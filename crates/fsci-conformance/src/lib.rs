#![forbid(unsafe_code)]

pub mod ci_gates;
pub mod dashboard;
pub mod e2e;
pub mod forensics;
pub mod quality_gates;

use asupersync::raptorq::systematic::SystematicEncoder;
use blake3::hash;
use fsci_arrayapi::{
    ArangeRequest as ArrayApiArangeRequest, ArrayApiArray as FsciArrayApiArray,
    CoreArray as FsciArrayApiCoreArray, CoreArrayBackend as FsciArrayApiCoreArrayBackend,
    CreationRequest as ArrayApiCreationRequest, DType as FsciArrayApiDType,
    ExecutionMode as FsciArrayApiExecutionMode, FullRequest as ArrayApiFullRequest,
    IndexExpr as FsciArrayApiIndexExpr, IndexRequest as ArrayApiIndexRequest,
    IndexingMode as FsciArrayApiIndexingMode, LinspaceRequest as ArrayApiLinspaceRequest,
    MemoryOrder as FsciArrayApiMemoryOrder, ScalarValue as FsciArrayApiScalarValue,
    Shape as FsciArrayApiShape, SliceSpec as FsciArrayApiSliceSpec, arange as arrayapi_arange,
    broadcast_shapes as arrayapi_broadcast_shapes, from_slice as arrayapi_from_slice,
    full as arrayapi_full, getitem as arrayapi_getitem, linspace as arrayapi_linspace,
    ones as arrayapi_ones, reshape as arrayapi_reshape, result_type as arrayapi_result_type,
    transpose as arrayapi_transpose, zeros as arrayapi_zeros,
};
use fsci_integrate::{ToleranceValue, validate_tol};
use fsci_linalg::{
    InvOptions, LinalgError, LstsqDriver, LstsqOptions, MatrixAssumption, PinvOptions,
    SolveOptions, TriangularSolveOptions, TriangularTranspose, det, inv, lstsq, pinv, solve,
    solve_banded, solve_triangular,
};
use fsci_opt::{
    ConvergenceStatus as OptConvergenceStatus, MinimizeOptions, OptError, OptimizeMethod,
    RootMethod, RootOptions, minimize, root_scalar,
};
use fsci_runtime::RuntimeMode;
use fsci_special::{
    SpecialError as FsciSpecialError, SpecialErrorKind as FsciSpecialErrorKind,
    SpecialTensor as FsciSpecialTensor, beta as special_beta, betainc as special_betainc,
    betaln as special_betaln, erf as special_erf, erfc as special_erfc, erfcinv as special_erfcinv,
    erfinv as special_erfinv, gamma as special_gamma, gammainc as special_gammainc,
    gammaincc as special_gammaincc, gammaln as special_gammaln, j0 as special_j0, j1 as special_j1,
    jn as special_jn, y0 as special_y0, y1 as special_y1, yn as special_yn,
};
#[cfg(feature = "dashboard")]
use ftui::{PackedRgba, Style};
use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub oracle_root: PathBuf,
    pub fixture_root: PathBuf,
    pub strict_mode: bool,
}

impl HarnessConfig {
    #[must_use]
    pub fn default_paths() -> Self {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        Self {
            oracle_root: repo_root.join("legacy_scipy_code/scipy"),
            fixture_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"),
            strict_mode: true,
        }
    }

    #[must_use]
    pub fn artifact_dir_for(&self, packet_id: &str) -> PathBuf {
        self.fixture_root.join("artifacts").join(packet_id)
    }
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self::default_paths()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessReport {
    pub suite: &'static str,
    pub oracle_present: bool,
    pub fixture_count: usize,
    pub strict_mode: bool,
}

#[derive(Debug, Error)]
pub enum HarnessError {
    #[error("fixture load failed for {path}: {source}")]
    FixtureIo { path: PathBuf, source: io::Error },
    #[error("fixture parse failed for {path}: {source}")]
    FixtureParse {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("artifact write failed for {path}: {source}")]
    ArtifactIo { path: PathBuf, source: io::Error },
    #[error("raptorq sidecar generation failed: {0}")]
    RaptorQ(String),
    #[error("linalg execution failed: {0}")]
    Linalg(#[from] LinalgError),
    #[error("optimize execution failed: {0}")]
    Optimize(#[from] OptError),
    #[error("failed to launch python oracle `{python_bin}`: {source}")]
    PythonLaunch {
        python_bin: String,
        source: io::Error,
    },
    #[error("python oracle script not found at {path}")]
    PythonScriptMissing { path: PathBuf },
    #[error("python oracle `{python_bin}` failed: {stderr}")]
    PythonFailed { python_bin: String, stderr: String },
    #[error("python oracle requires scipy but it is unavailable: {stderr}")]
    PythonSciPyMissing { stderr: String },
    #[error("oracle capture parse failed for {path}: {source}")]
    OracleParse {
        path: PathBuf,
        source: serde_json::Error,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum FixtureToleranceValue {
    Scalar(f64),
    Vector(Vec<f64>),
}

impl From<FixtureToleranceValue> for ToleranceValue {
    fn from(value: FixtureToleranceValue) -> Self {
        match value {
            FixtureToleranceValue::Scalar(v) => Self::Scalar(v),
            FixtureToleranceValue::Vector(vs) => Self::Vector(vs),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExpectedOutcome {
    Ok {
        rtol: FixtureToleranceValue,
        atol: FixtureToleranceValue,
        warning_rtol_clamped: bool,
    },
    Error {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidateTolCase {
    pub case_id: String,
    pub mode: RuntimeMode,
    pub n: usize,
    pub rtol: FixtureToleranceValue,
    pub atol: FixtureToleranceValue,
    pub expected: ExpectedOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<ValidateTolCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FixtureMatrixAssumption {
    General,
    Diagonal,
    UpperTriangular,
    LowerTriangular,
    Symmetric,
    Hermitian,
    PositiveDefinite,
}

impl From<FixtureMatrixAssumption> for MatrixAssumption {
    fn from(value: FixtureMatrixAssumption) -> Self {
        match value {
            FixtureMatrixAssumption::General => Self::General,
            FixtureMatrixAssumption::Diagonal => Self::Diagonal,
            FixtureMatrixAssumption::UpperTriangular => Self::UpperTriangular,
            FixtureMatrixAssumption::LowerTriangular => Self::LowerTriangular,
            FixtureMatrixAssumption::Symmetric => Self::Symmetric,
            FixtureMatrixAssumption::Hermitian => Self::Hermitian,
            FixtureMatrixAssumption::PositiveDefinite => Self::PositiveDefinite,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FixtureTriangularTranspose {
    NoTranspose,
    Transpose,
    ConjugateTranspose,
}

impl From<FixtureTriangularTranspose> for TriangularTranspose {
    fn from(value: FixtureTriangularTranspose) -> Self {
        match value {
            FixtureTriangularTranspose::NoTranspose => Self::NoTranspose,
            FixtureTriangularTranspose::Transpose => Self::Transpose,
            FixtureTriangularTranspose::ConjugateTranspose => Self::ConjugateTranspose,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LinalgExpectedOutcome {
    Vector {
        values: Vec<f64>,
        atol: f64,
        rtol: f64,
        #[serde(default)]
        expect_warning_ill_conditioned: Option<bool>,
    },
    Matrix {
        values: Vec<Vec<f64>>,
        atol: f64,
        rtol: f64,
    },
    Scalar {
        value: f64,
        atol: f64,
        rtol: f64,
    },
    Lstsq {
        x: Vec<f64>,
        residuals: Vec<f64>,
        rank: usize,
        singular_values: Vec<f64>,
        atol: f64,
        rtol: f64,
    },
    Pinv {
        values: Vec<Vec<f64>>,
        rank: usize,
        atol: f64,
        rtol: f64,
    },
    Error {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum LinalgCase {
    Solve {
        case_id: String,
        mode: RuntimeMode,
        a: Vec<Vec<f64>>,
        b: Vec<f64>,
        assume_a: Option<FixtureMatrixAssumption>,
        #[serde(default)]
        lower: Option<bool>,
        #[serde(default)]
        transposed: Option<bool>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    SolveTriangular {
        case_id: String,
        mode: RuntimeMode,
        a: Vec<Vec<f64>>,
        b: Vec<f64>,
        trans: Option<FixtureTriangularTranspose>,
        #[serde(default)]
        lower: Option<bool>,
        #[serde(default)]
        unit_diagonal: Option<bool>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    SolveBanded {
        case_id: String,
        mode: RuntimeMode,
        l_and_u: [usize; 2],
        ab: Vec<Vec<f64>>,
        b: Vec<f64>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Inv {
        case_id: String,
        mode: RuntimeMode,
        a: Vec<Vec<f64>>,
        assume_a: Option<FixtureMatrixAssumption>,
        #[serde(default)]
        lower: Option<bool>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Det {
        case_id: String,
        mode: RuntimeMode,
        a: Vec<Vec<f64>>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Lstsq {
        case_id: String,
        mode: RuntimeMode,
        a: Vec<Vec<f64>>,
        b: Vec<f64>,
        #[serde(default)]
        cond: Option<f64>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Pinv {
        case_id: String,
        mode: RuntimeMode,
        a: Vec<Vec<f64>>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
}

impl LinalgCase {
    fn case_id(&self) -> &str {
        match self {
            Self::Solve { case_id, .. }
            | Self::SolveTriangular { case_id, .. }
            | Self::SolveBanded { case_id, .. }
            | Self::Inv { case_id, .. }
            | Self::Det { case_id, .. }
            | Self::Lstsq { case_id, .. }
            | Self::Pinv { case_id, .. } => case_id,
        }
    }

    fn expected(&self) -> &LinalgExpectedOutcome {
        match self {
            Self::Solve { expected, .. }
            | Self::SolveTriangular { expected, .. }
            | Self::SolveBanded { expected, .. }
            | Self::Inv { expected, .. }
            | Self::Det { expected, .. }
            | Self::Lstsq { expected, .. }
            | Self::Pinv { expected, .. } => expected,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LinalgPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<LinalgCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizeMinObjective {
    Rosenbrock2,
    Ackley2,
    Rastrigin2,
    Sphere2,
    ShiftedQuadratic,
    TranslatedQuadratic,
    ScaledQuadratic,
    RotatedQuadratic,
    L1Nonsmooth,
    FlatQuartic,
    NanBranch,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizeRootObjective {
    CubicMinusTwo,
    CosMinusX,
    SinMinusHalf,
    LinearShift03,
    NanBranch,
    StepDiscontinuous,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OptimizeExpectedOutcome {
    MinimizePoint {
        x: Vec<f64>,
        fun: f64,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        require_success: Option<bool>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    RootValue {
        root: f64,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        require_converged: Option<bool>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    MinimizeStatus {
        status: OptConvergenceStatus,
        success: bool,
    },
    RootStatus {
        status: OptConvergenceStatus,
        converged: bool,
    },
    Error {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum OptimizeCase {
    Minimize {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        method: OptimizeMethod,
        objective: OptimizeMinObjective,
        x0: Vec<f64>,
        #[serde(default)]
        tol: Option<f64>,
        #[serde(default)]
        maxiter: Option<usize>,
        #[serde(default)]
        maxfev: Option<usize>,
        expected: OptimizeExpectedOutcome,
    },
    Root {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        method: RootMethod,
        objective: OptimizeRootObjective,
        bracket: [f64; 2],
        #[serde(default)]
        xtol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        maxiter: Option<usize>,
        expected: OptimizeExpectedOutcome,
    },
}

impl OptimizeCase {
    fn case_id(&self) -> &str {
        match self {
            Self::Minimize { case_id, .. } | Self::Root { case_id, .. } => case_id,
        }
    }

    #[cfg(test)]
    fn category(&self) -> &str {
        match self {
            Self::Minimize { category, .. } | Self::Root { category, .. } => category,
        }
    }

    fn expected(&self) -> &OptimizeExpectedOutcome {
        match self {
            Self::Minimize { expected, .. } | Self::Root { expected, .. } => expected,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OptimizePacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<OptimizeCase>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SpecialCaseFunction {
    Gamma,
    Gammaln,
    Gammainc,
    Gammaincc,
    Erf,
    Erfc,
    Erfinv,
    Erfcinv,
    Beta,
    Betaln,
    Betainc,
    J0,
    J1,
    Jn,
    Y0,
    Y1,
    Yn,
    RelErfErfcIdentity,
    RelGammaRecurrence,
    RelBetaSymmetry,
    RelGammaincComplement,
    RelJnRecurrence,
    RelErfinvComposition,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpecialValueClass {
    Finite,
    Nan,
    PosInf,
    NegInf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpecialExpectedOutcome {
    Scalar {
        value: f64,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    Class {
        class: SpecialValueClass,
    },
    ErrorKind {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpecialCase {
    pub case_id: String,
    pub category: String,
    pub mode: RuntimeMode,
    pub function: SpecialCaseFunction,
    #[serde(default)]
    pub args: Vec<f64>,
    pub expected: SpecialExpectedOutcome,
}

impl SpecialCase {
    fn case_id(&self) -> &str {
        &self.case_id
    }

    #[cfg(test)]
    fn category(&self) -> &str {
        &self.category
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpecialPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<SpecialCase>,
}

// ---------- Sparse packet fixture types ----------

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SparseOperation {
    Spmv,
    Spsolve,
    FormatRoundtrip,
    Add,
    Scale,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SparseInputFormat {
    Coo,
    Csr,
    Csc,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SparseExpectedOutcome {
    Vector {
        values: Vec<f64>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
    },
    Scalar {
        value: f64,
        #[serde(default)]
        atol: Option<f64>,
    },
    Shape {
        rows: usize,
        cols: usize,
        nnz: usize,
    },
    Error {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SparseCase {
    pub case_id: String,
    pub category: String,
    pub mode: RuntimeMode,
    pub operation: SparseOperation,
    pub format: SparseInputFormat,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    #[serde(default)]
    pub rhs: Option<Vec<f64>>,
    #[serde(default)]
    pub scalar: Option<f64>,
    pub expected: SparseExpectedOutcome,
}

impl SparseCase {
    fn case_id(&self) -> &str {
        &self.case_id
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SparsePacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<SparseCase>,
}

// ── FFT fixture types ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FftTransformKind {
    Fft,
    Ifft,
    Rfft,
    Irfft,
    Fft2,
    Ifft2,
    Fftn,
    Fftfreq,
    Rfftfreq,
    Fftshift,
    Ifftshift,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FftNormalization {
    Forward,
    Backward,
    Ortho,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FftExpectedOutcome {
    ComplexVector {
        values: Vec<[f64; 2]>,
        atol: Option<f64>,
        rtol: Option<f64>,
    },
    RealVector {
        values: Vec<f64>,
        atol: Option<f64>,
        rtol: Option<f64>,
    },
    Error {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FftCase {
    pub case_id: String,
    pub category: String,
    pub mode: RuntimeMode,
    pub transform: FftTransformKind,
    pub normalization: Option<FftNormalization>,
    /// Real input (for rfft, irfft output, fftfreq, rfftfreq, fftshift, ifftshift)
    pub real_input: Option<Vec<f64>>,
    /// Complex input as [[re, im], ...] (for fft, ifft, irfft input)
    pub complex_input: Option<Vec<[f64; 2]>>,
    /// For irfft: desired output length
    pub output_len: Option<usize>,
    /// For fftfreq/rfftfreq: sample spacing
    pub sample_spacing: Option<f64>,
    /// For 2D/ND transforms: shape
    pub shape: Option<Vec<usize>>,
    pub expected: FftExpectedOutcome,
}

impl FftCase {
    fn case_id(&self) -> &str {
        &self.case_id
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FftPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<FftCase>,
}

// ── CASP Runtime fixture types ───────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CaspTestKind {
    PolicyDecision,
    SolverSelection,
    CalibratorDrift,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CaspExpectedOutcome {
    PolicyAction { action: String },
    SolverAction { action: String },
    CalibratorFallback { should_fallback: bool },
    Error { error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaspCase {
    pub case_id: String,
    pub category: String,
    pub mode: RuntimeMode,
    pub test_kind: CaspTestKind,
    /// For policy decisions: condition_signal, metadata_signal, anomaly_signal
    pub condition_signal: Option<f64>,
    pub metadata_signal: Option<f64>,
    pub anomaly_signal: Option<f64>,
    /// For solver selection: matrix condition state
    pub condition_state: Option<String>,
    /// For calibrator: sequence of backward error observations
    pub observations: Option<Vec<f64>>,
    pub expected: CaspExpectedOutcome,
}

impl CaspCase {
    fn case_id(&self) -> &str {
        &self.case_id
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaspPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<CaspCase>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArrayApiFixtureDType {
    Bool,
    Int64,
    UInt64,
    Float32,
    Float64,
    Complex64,
    Complex128,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArrayApiFixtureScalar {
    Bool { value: bool },
    I64 { value: i64 },
    U64 { value: u64 },
    F64 { value: f64 },
    ComplexF64 { re: f64, im: f64 },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArrayApiFixtureIndexingMode {
    Basic,
    Advanced,
    BooleanMask,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArrayApiFixtureSliceSpec {
    pub start: Option<isize>,
    pub stop: Option<isize>,
    pub step: isize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArrayApiFixtureIndexExpr {
    Basic {
        slices: Vec<ArrayApiFixtureSliceSpec>,
    },
    Advanced {
        indices: Vec<Vec<isize>>,
    },
    BooleanMask {
        mask_shape: Vec<usize>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArrayApiExpectedOutcome {
    Array {
        shape: Vec<usize>,
        dtype: ArrayApiFixtureDType,
        values: Vec<ArrayApiFixtureScalar>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    Shape {
        dims: Vec<usize>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    Dtype {
        dtype: ArrayApiFixtureDType,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    Bool {
        value: bool,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    ErrorKind {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum ArrayApiCase {
    Zeros {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        shape: Vec<usize>,
        dtype: ArrayApiFixtureDType,
        expected: ArrayApiExpectedOutcome,
    },
    Ones {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        shape: Vec<usize>,
        dtype: ArrayApiFixtureDType,
        expected: ArrayApiExpectedOutcome,
    },
    Full {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        shape: Vec<usize>,
        fill_value: ArrayApiFixtureScalar,
        dtype: ArrayApiFixtureDType,
        expected: ArrayApiExpectedOutcome,
    },
    Arange {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        start: ArrayApiFixtureScalar,
        stop: ArrayApiFixtureScalar,
        step: ArrayApiFixtureScalar,
        #[serde(default)]
        dtype: Option<ArrayApiFixtureDType>,
        expected: ArrayApiExpectedOutcome,
    },
    Linspace {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        start: ArrayApiFixtureScalar,
        stop: ArrayApiFixtureScalar,
        num: usize,
        endpoint: bool,
        #[serde(default)]
        dtype: Option<ArrayApiFixtureDType>,
        expected: ArrayApiExpectedOutcome,
    },
    BroadcastShapes {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        shapes: Vec<Vec<usize>>,
        expected: ArrayApiExpectedOutcome,
    },
    ResultType {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        dtypes: Vec<ArrayApiFixtureDType>,
        #[serde(default)]
        force_floating: bool,
        expected: ArrayApiExpectedOutcome,
    },
    FromSlice {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        values: Vec<ArrayApiFixtureScalar>,
        shape: Vec<usize>,
        dtype: ArrayApiFixtureDType,
        expected: ArrayApiExpectedOutcome,
    },
    Getitem {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        source_values: Vec<ArrayApiFixtureScalar>,
        source_shape: Vec<usize>,
        source_dtype: ArrayApiFixtureDType,
        indexing_mode: ArrayApiFixtureIndexingMode,
        index: ArrayApiFixtureIndexExpr,
        expected: ArrayApiExpectedOutcome,
    },
    Reshape {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        source_values: Vec<ArrayApiFixtureScalar>,
        source_shape: Vec<usize>,
        source_dtype: ArrayApiFixtureDType,
        new_shape: Vec<usize>,
        expected: ArrayApiExpectedOutcome,
    },
    Transpose {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        source_values: Vec<ArrayApiFixtureScalar>,
        source_shape: Vec<usize>,
        source_dtype: ArrayApiFixtureDType,
        expected: ArrayApiExpectedOutcome,
    },
    RelationBroadcastCommutative {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
        expected: ArrayApiExpectedOutcome,
    },
    RelationResultTypeSymmetry {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        left_dtype: ArrayApiFixtureDType,
        right_dtype: ArrayApiFixtureDType,
        #[serde(default)]
        force_floating: bool,
        expected: ArrayApiExpectedOutcome,
    },
    RelationIndexRoundtrip {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        values: Vec<ArrayApiFixtureScalar>,
        dtype: ArrayApiFixtureDType,
        index: isize,
        expected: ArrayApiExpectedOutcome,
    },
}

impl ArrayApiCase {
    fn case_id(&self) -> &str {
        match self {
            Self::Zeros { case_id, .. }
            | Self::Ones { case_id, .. }
            | Self::Full { case_id, .. }
            | Self::Arange { case_id, .. }
            | Self::Linspace { case_id, .. }
            | Self::BroadcastShapes { case_id, .. }
            | Self::ResultType { case_id, .. }
            | Self::FromSlice { case_id, .. }
            | Self::Getitem { case_id, .. }
            | Self::Reshape { case_id, .. }
            | Self::Transpose { case_id, .. }
            | Self::RelationBroadcastCommutative { case_id, .. }
            | Self::RelationResultTypeSymmetry { case_id, .. }
            | Self::RelationIndexRoundtrip { case_id, .. } => case_id,
        }
    }

    #[cfg(test)]
    fn category(&self) -> &str {
        match self {
            Self::Zeros { category, .. }
            | Self::Ones { category, .. }
            | Self::Full { category, .. }
            | Self::Arange { category, .. }
            | Self::Linspace { category, .. }
            | Self::BroadcastShapes { category, .. }
            | Self::ResultType { category, .. }
            | Self::FromSlice { category, .. }
            | Self::Getitem { category, .. }
            | Self::Reshape { category, .. }
            | Self::Transpose { category, .. }
            | Self::RelationBroadcastCommutative { category, .. }
            | Self::RelationResultTypeSymmetry { category, .. }
            | Self::RelationIndexRoundtrip { category, .. } => category,
        }
    }

    fn expected(&self) -> &ArrayApiExpectedOutcome {
        match self {
            Self::Zeros { expected, .. }
            | Self::Ones { expected, .. }
            | Self::Full { expected, .. }
            | Self::Arange { expected, .. }
            | Self::Linspace { expected, .. }
            | Self::BroadcastShapes { expected, .. }
            | Self::ResultType { expected, .. }
            | Self::FromSlice { expected, .. }
            | Self::Getitem { expected, .. }
            | Self::Reshape { expected, .. }
            | Self::Transpose { expected, .. }
            | Self::RelationBroadcastCommutative { expected, .. }
            | Self::RelationResultTypeSymmetry { expected, .. }
            | Self::RelationIndexRoundtrip { expected, .. } => expected,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArrayApiPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<ArrayApiCase>,
}

#[derive(Debug, Clone, PartialEq)]
enum ArrayApiObservedOutcome {
    Array {
        shape: Vec<usize>,
        dtype: ArrayApiFixtureDType,
        values: Vec<ArrayApiFixtureScalar>,
    },
    Shape {
        dims: Vec<usize>,
    },
    Dtype {
        dtype: ArrayApiFixtureDType,
    },
    Bool {
        value: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaseResult {
    pub case_id: String,
    pub passed: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PacketReport {
    pub packet_id: String,
    pub family: String,
    pub case_results: Vec<CaseResult>,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub generated_unix_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PacketSummary {
    pub packet_id: String,
    pub family: String,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub total_cases: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RaptorQSidecar {
    pub schema_version: u8,
    pub source_hash: String,
    pub symbol_size: usize,
    pub source_symbols: usize,
    pub repair_symbols: usize,
    pub repair_symbol_hashes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecodeProofArtifact {
    pub ts_unix_ms: u128,
    pub reason: String,
    pub recovered_blocks: usize,
    pub proof_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParityArtifactBundle {
    pub report_path: PathBuf,
    pub sidecar_path: PathBuf,
    pub decode_proof_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OracleCapture {
    pub packet_id: String,
    pub family: String,
    pub generated_unix_ms: u128,
    pub case_outputs: Vec<OracleCaseOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OracleCaseOutput {
    pub case_id: String,
    pub status: String,
    pub result_kind: String,
    pub result: serde_json::Value,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PythonOracleConfig {
    pub python_bin: PathBuf,
    pub script_path: PathBuf,
    pub required: bool,
}

impl Default for PythonOracleConfig {
    fn default() -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            python_bin: PathBuf::from("python3"),
            script_path: manifest.join("python_oracle/scipy_linalg_oracle.py"),
            required: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum LinalgObservedOutcome {
    Vector {
        values: Vec<f64>,
        warning_ill_conditioned: bool,
    },
    Matrix {
        values: Vec<Vec<f64>>,
    },
    Scalar {
        value: f64,
    },
    Lstsq {
        x: Vec<f64>,
        residuals: Vec<f64>,
        rank: usize,
        singular_values: Vec<f64>,
    },
    Pinv {
        values: Vec<Vec<f64>>,
        rank: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum OptimizeObservedOutcome {
    Minimize {
        x: Vec<f64>,
        fun: Option<f64>,
        success: bool,
        status: OptConvergenceStatus,
    },
    Root {
        root: f64,
        converged: bool,
        status: OptConvergenceStatus,
    },
}

#[must_use]
pub fn run_smoke(config: &HarnessConfig) -> HarnessReport {
    let fixture_count = fs::read_dir(&config.fixture_root)
        .ok()
        .into_iter()
        .flat_map(|it| it.filter_map(Result::ok))
        .count();

    HarnessReport {
        suite: "smoke",
        oracle_present: config.oracle_root.exists(),
        fixture_count,
        strict_mode: config.strict_mode,
    }
}

pub fn run_validate_tol_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: PacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let outcome = validate_tol(
            case.rtol.clone().into(),
            case.atol.clone().into(),
            case.n,
            case.mode,
        );

        let (passed, message) = match (&case.expected, outcome) {
            (
                ExpectedOutcome::Ok {
                    rtol,
                    atol,
                    warning_rtol_clamped,
                },
                Ok(actual),
            ) => {
                let actual_warning = !actual.warnings.is_empty();
                let expected_rtol: ToleranceValue = rtol.clone().into();
                let expected_atol: ToleranceValue = atol.clone().into();
                let pass = actual.rtol == expected_rtol
                    && actual.atol == expected_atol
                    && actual_warning == *warning_rtol_clamped;
                let msg = if pass {
                    "output matched expected tolerance contract".to_owned()
                } else {
                    format!(
                        "mismatch: expected rtol={expected_rtol:?}, atol={expected_atol:?}, warning={warning_rtol_clamped}; got rtol={:?}, atol={:?}, warning={actual_warning}",
                        actual.rtol, actual.atol
                    )
                };
                (pass, msg)
            }
            (ExpectedOutcome::Error { error }, Err(actual)) => {
                let pass = error == &actual.to_string();
                let msg = if pass {
                    "error matched expected contract".to_owned()
                } else {
                    format!("mismatch: expected error `{error}`, got `{actual}`")
                };
                (pass, msg)
            }
            (expected, result) => (
                false,
                format!("shape mismatch: expected {expected:?}, got {result:?}"),
            ),
        };

        case_results.push(CaseResult {
            case_id: case.case_id.clone(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

pub fn run_linalg_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: LinalgPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = execute_linalg_case(case);
        let (passed, message) = compare_linalg_case(case.expected(), &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

pub fn run_optimize_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: OptimizePacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = execute_optimize_case(case);
        let (passed, message, _, _) =
            compare_optimize_case_differential(case.expected(), &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

pub fn run_special_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: SpecialPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = execute_special_case(case);
        let (passed, message, _, _) = compare_special_case_differential(case, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

pub fn run_array_api_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: ArrayApiPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = execute_array_api_case(case);
        let (passed, message, _, _) = compare_array_api_case_differential(case, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

pub fn run_sparse_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: SparsePacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = execute_sparse_case(case);
        let (passed, message) = compare_sparse_outcome(&case.expected, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

#[derive(Debug)]
enum SparseObserved {
    Vector(Vec<f64>),
    Shape {
        rows: usize,
        cols: usize,
        nnz: usize,
    },
    Error(String),
}

fn execute_sparse_case(case: &SparseCase) -> SparseObserved {
    use fsci_sparse::{CooMatrix, FormatConvertible, Shape2D, SolveOptions};

    let shape = Shape2D::new(case.rows, case.cols);

    let coo = match CooMatrix::from_triplets(
        shape,
        case.data.clone(),
        case.row_indices.clone(),
        case.col_indices.clone(),
        true,
    ) {
        Ok(m) => m,
        Err(e) => return SparseObserved::Error(format!("{e}")),
    };

    match case.operation {
        SparseOperation::Spmv => {
            let rhs = match &case.rhs {
                Some(v) => v.as_slice(),
                None => return SparseObserved::Error("spmv requires rhs".to_owned()),
            };
            match case.format {
                SparseInputFormat::Coo => match fsci_sparse::spmv_coo(&coo, rhs) {
                    Ok(v) => SparseObserved::Vector(v),
                    Err(e) => SparseObserved::Error(format!("{e}")),
                },
                SparseInputFormat::Csr => {
                    let csr = match coo.to_csr() {
                        Ok(c) => c,
                        Err(e) => return SparseObserved::Error(format!("{e}")),
                    };
                    match fsci_sparse::spmv_csr(&csr, rhs) {
                        Ok(v) => SparseObserved::Vector(v),
                        Err(e) => SparseObserved::Error(format!("{e}")),
                    }
                }
                SparseInputFormat::Csc => {
                    let csc = match coo.to_csc() {
                        Ok(c) => c,
                        Err(e) => return SparseObserved::Error(format!("{e}")),
                    };
                    match fsci_sparse::spmv_csc(&csc, rhs) {
                        Ok(v) => SparseObserved::Vector(v),
                        Err(e) => SparseObserved::Error(format!("{e}")),
                    }
                }
            }
        }
        SparseOperation::Spsolve => {
            let rhs = match &case.rhs {
                Some(v) => v.as_slice(),
                None => return SparseObserved::Error("spsolve requires rhs".to_owned()),
            };
            let csr = match coo.to_csr() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            match fsci_sparse::spsolve(&csr, rhs, SolveOptions::default()) {
                Ok(result) => SparseObserved::Vector(result.solution),
                Err(e) => SparseObserved::Error(format!("{e}")),
            }
        }
        SparseOperation::FormatRoundtrip => {
            let csr = match coo.to_csr() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            let csc = match csr.to_csc() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            // Verify CSC→CSR roundtrip preserves structure
            let csr2 = match csc.to_csr() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            SparseObserved::Shape {
                rows: csr2.shape().rows,
                cols: csr2.shape().cols,
                nnz: csr2.nnz(),
            }
        }
        SparseOperation::Add => {
            let csr_a = match coo.to_csr() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            match fsci_sparse::add_csr(&csr_a, &csr_a) {
                Ok(result) => {
                    let rhs = match &case.rhs {
                        Some(v) => v.as_slice(),
                        None => {
                            return SparseObserved::Shape {
                                rows: result.shape().rows,
                                cols: result.shape().cols,
                                nnz: result.nnz(),
                            };
                        }
                    };
                    match fsci_sparse::spmv_csr(&result, rhs) {
                        Ok(v) => SparseObserved::Vector(v),
                        Err(e) => SparseObserved::Error(format!("{e}")),
                    }
                }
                Err(e) => SparseObserved::Error(format!("{e}")),
            }
        }
        SparseOperation::Scale => {
            let csr = match coo.to_csr() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            let scalar = case.scalar.unwrap_or(1.0);
            match fsci_sparse::scale_csr(&csr, scalar) {
                Ok(result) => {
                    let rhs = match &case.rhs {
                        Some(v) => v.as_slice(),
                        None => {
                            return SparseObserved::Shape {
                                rows: result.shape().rows,
                                cols: result.shape().cols,
                                nnz: result.nnz(),
                            };
                        }
                    };
                    match fsci_sparse::spmv_csr(&result, rhs) {
                        Ok(v) => SparseObserved::Vector(v),
                        Err(e) => SparseObserved::Error(format!("{e}")),
                    }
                }
                Err(e) => SparseObserved::Error(format!("{e}")),
            }
        }
    }
}

fn compare_sparse_outcome(
    expected: &SparseExpectedOutcome,
    observed: &SparseObserved,
) -> (bool, String) {
    match (expected, observed) {
        (SparseExpectedOutcome::Vector { values, atol, rtol }, SparseObserved::Vector(got)) => {
            let a = atol.unwrap_or(1.0e-10);
            let r = rtol.unwrap_or(1.0e-8);
            if got.len() != values.len() {
                return (
                    false,
                    format!(
                        "vector length mismatch: expected {} got {}",
                        values.len(),
                        got.len()
                    ),
                );
            }
            let max_diff = got
                .iter()
                .zip(values.iter())
                .map(|(g, e)| (g - e).abs())
                .fold(0.0_f64, f64::max);
            let pass = got
                .iter()
                .zip(values.iter())
                .all(|(g, e)| allclose_scalar(*g, *e, a, r));
            if pass {
                (true, format!("vector matched (max_diff={max_diff:.2e})"))
            } else {
                (
                    false,
                    format!("vector mismatch: max_diff={max_diff:.2e}, atol={a:.2e}, rtol={r:.2e}"),
                )
            }
        }
        (
            SparseExpectedOutcome::Shape { rows, cols, nnz },
            SparseObserved::Shape {
                rows: gr,
                cols: gc,
                nnz: gn,
            },
        ) => {
            let pass = *rows == *gr && *cols == *gc && *nnz == *gn;
            if pass {
                (true, format!("shape matched ({gr}x{gc}, nnz={gn})"))
            } else {
                (
                    false,
                    format!(
                        "shape mismatch: expected {rows}x{cols} nnz={nnz}, got {gr}x{gc} nnz={gn}"
                    ),
                )
            }
        }
        (SparseExpectedOutcome::Error { error }, SparseObserved::Error(got)) => {
            let pass = got.contains(error.as_str());
            if pass {
                (true, format!("error matched: {got}"))
            } else {
                (
                    false,
                    format!("error mismatch: expected '{error}', got '{got}'"),
                )
            }
        }
        (SparseExpectedOutcome::Error { error }, _) => {
            (false, format!("expected error '{error}' but got success"))
        }
        (_, SparseObserved::Error(e)) => (false, format!("unexpected error: {e}")),
        _ => (false, "outcome type mismatch".to_owned()),
    }
}

// ── FFT packet runner ────────────────────────────────────────────────

pub fn run_fft_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: FftPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = execute_fft_case(case);
        let (passed, message) = compare_fft_outcome(&case.expected, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

#[derive(Debug)]
enum FftObserved {
    ComplexVector(Vec<[f64; 2]>),
    RealVector(Vec<f64>),
    Error(String),
}

fn execute_fft_case(case: &FftCase) -> FftObserved {
    use fsci_fft::{FftOptions, Normalization};

    let mut opts = FftOptions::default().with_mode(case.mode);
    if let Some(norm) = &case.normalization {
        opts = opts.with_normalization(match norm {
            FftNormalization::Forward => Normalization::Forward,
            FftNormalization::Backward => Normalization::Backward,
            FftNormalization::Ortho => Normalization::Ortho,
        });
    }

    match case.transform {
        FftTransformKind::Fft => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("fft requires complex_input".to_owned()),
            };
            match fsci_fft::fft(&input, &opts) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Ifft => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("ifft requires complex_input".to_owned()),
            };
            match fsci_fft::ifft(&input, &opts) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Rfft => {
            let input = match &case.real_input {
                Some(v) => v.as_slice(),
                None => return FftObserved::Error("rfft requires real_input".to_owned()),
            };
            match fsci_fft::rfft(input, &opts) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Irfft => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("irfft requires complex_input".to_owned()),
            };
            match fsci_fft::irfft(&input, case.output_len, &opts) {
                Ok(v) => FftObserved::RealVector(v),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Fftfreq => {
            let n = case.real_input.as_ref().map_or(0, |v| v.len());
            let spacing = case.sample_spacing.unwrap_or(1.0);
            match fsci_fft::fftfreq(n, spacing) {
                Ok(v) => FftObserved::RealVector(v),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Rfftfreq => {
            let n = case.real_input.as_ref().map_or(0, |v| v.len());
            let spacing = case.sample_spacing.unwrap_or(1.0);
            match fsci_fft::rfftfreq(n, spacing) {
                Ok(v) => FftObserved::RealVector(v),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Fftshift => {
            let input = match &case.real_input {
                Some(v) => v.clone(),
                None => return FftObserved::Error("fftshift requires real_input".to_owned()),
            };
            FftObserved::RealVector(fsci_fft::fftshift_1d(&input))
        }
        FftTransformKind::Ifftshift => {
            let input = match &case.real_input {
                Some(v) => v.clone(),
                None => return FftObserved::Error("ifftshift requires real_input".to_owned()),
            };
            FftObserved::RealVector(fsci_fft::ifftshift_1d(&input))
        }
        FftTransformKind::Fft2 | FftTransformKind::Ifft2 | FftTransformKind::Fftn => {
            // Multi-dimensional transforms: require complex_input and shape
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => {
                    return FftObserved::Error("nd transforms require complex_input".to_owned());
                }
            };
            let shape = match &case.shape {
                Some(s) => s.clone(),
                None => return FftObserved::Error("nd transforms require shape".to_owned()),
            };
            let result = match case.transform {
                FftTransformKind::Fft2 => {
                    if shape.len() != 2 {
                        return FftObserved::Error("fft2 requires 2D shape".to_owned());
                    }
                    fsci_fft::fft2(&input, (shape[0], shape[1]), &opts)
                }
                FftTransformKind::Ifft2 => {
                    if shape.len() != 2 {
                        return FftObserved::Error("ifft2 requires 2D shape".to_owned());
                    }
                    fsci_fft::ifft2(&input, (shape[0], shape[1]), &opts)
                }
                FftTransformKind::Fftn => fsci_fft::fftn(&input, &shape, &opts),
                _ => unreachable!(),
            };
            match result {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
    }
}

fn compare_fft_outcome(expected: &FftExpectedOutcome, observed: &FftObserved) -> (bool, String) {
    match (expected, observed) {
        (
            FftExpectedOutcome::ComplexVector { values, atol, rtol },
            FftObserved::ComplexVector(got),
        ) => {
            let a = atol.unwrap_or(1.0e-10);
            let r = rtol.unwrap_or(1.0e-8);
            if got.len() != values.len() {
                return (
                    false,
                    format!(
                        "complex vector length mismatch: expected {} got {}",
                        values.len(),
                        got.len()
                    ),
                );
            }
            let max_diff = got
                .iter()
                .zip(values.iter())
                .map(|(g, e)| {
                    let dr = (g[0] - e[0]).abs();
                    let di = (g[1] - e[1]).abs();
                    dr.max(di)
                })
                .fold(0.0_f64, f64::max);
            let pass = got.iter().zip(values.iter()).all(|(g, e)| {
                allclose_scalar(g[0], e[0], a, r) && allclose_scalar(g[1], e[1], a, r)
            });
            if pass {
                (
                    true,
                    format!("complex vector matched (max_diff={max_diff:.2e})"),
                )
            } else {
                (
                    false,
                    format!(
                        "complex vector mismatch: max_diff={max_diff:.2e}, atol={a:.2e}, rtol={r:.2e}"
                    ),
                )
            }
        }
        (FftExpectedOutcome::RealVector { values, atol, rtol }, FftObserved::RealVector(got)) => {
            let a = atol.unwrap_or(1.0e-10);
            let r = rtol.unwrap_or(1.0e-8);
            if got.len() != values.len() {
                return (
                    false,
                    format!(
                        "real vector length mismatch: expected {} got {}",
                        values.len(),
                        got.len()
                    ),
                );
            }
            let max_diff = got
                .iter()
                .zip(values.iter())
                .map(|(g, e)| (g - e).abs())
                .fold(0.0_f64, f64::max);
            let pass = got
                .iter()
                .zip(values.iter())
                .all(|(g, e)| allclose_scalar(*g, *e, a, r));
            if pass {
                (
                    true,
                    format!("real vector matched (max_diff={max_diff:.2e})"),
                )
            } else {
                (
                    false,
                    format!(
                        "real vector mismatch: max_diff={max_diff:.2e}, atol={a:.2e}, rtol={r:.2e}"
                    ),
                )
            }
        }
        (FftExpectedOutcome::Error { error }, FftObserved::Error(got)) => {
            let pass = got.contains(error.as_str());
            if pass {
                (true, format!("error matched: {got}"))
            } else {
                (
                    false,
                    format!("error mismatch: expected '{error}', got '{got}'"),
                )
            }
        }
        (FftExpectedOutcome::Error { error }, _) => {
            (false, format!("expected error '{error}' but got success"))
        }
        (_, FftObserved::Error(e)) => (false, format!("unexpected error: {e}")),
        _ => (false, "outcome type mismatch".to_owned()),
    }
}

// ── CASP Runtime packet runner ───────────────────────────────────────

pub fn run_casp_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: CaspPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = execute_casp_case(case);
        let (passed, message) = compare_casp_outcome(&case.expected, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

#[derive(Debug)]
enum CaspObserved {
    PolicyAction(String),
    SolverAction(String),
    CalibratorFallback(bool),
    Error(String),
}

fn parse_condition_state(s: &str) -> Option<fsci_runtime::MatrixConditionState> {
    match s {
        "well_conditioned" => Some(fsci_runtime::MatrixConditionState::WellConditioned),
        "moderate_condition" => Some(fsci_runtime::MatrixConditionState::ModerateCondition),
        "ill_conditioned" => Some(fsci_runtime::MatrixConditionState::IllConditioned),
        "near_singular" => Some(fsci_runtime::MatrixConditionState::NearSingular),
        _ => None,
    }
}

fn policy_action_name(action: &fsci_runtime::PolicyAction) -> &'static str {
    match action {
        fsci_runtime::PolicyAction::Allow => "allow",
        fsci_runtime::PolicyAction::FullValidate => "full_validate",
        fsci_runtime::PolicyAction::FailClosed => "fail_closed",
    }
}

fn solver_action_name(action: &fsci_runtime::SolverAction) -> &'static str {
    match action {
        fsci_runtime::SolverAction::DirectLU => "direct_lu",
        fsci_runtime::SolverAction::PivotedQR => "pivoted_qr",
        fsci_runtime::SolverAction::SVDFallback => "svd_fallback",
        fsci_runtime::SolverAction::DiagonalFastPath => "diagonal_fast_path",
        fsci_runtime::SolverAction::TriangularFastPath => "triangular_fast_path",
    }
}

fn execute_casp_case(case: &CaspCase) -> CaspObserved {
    match case.test_kind {
        CaspTestKind::PolicyDecision => {
            let cond = case.condition_signal.unwrap_or(0.0);
            let meta = case.metadata_signal.unwrap_or(0.0);
            let anom = case.anomaly_signal.unwrap_or(0.0);

            let signals = fsci_runtime::DecisionSignals::new(cond, meta, anom);
            let mut controller = fsci_runtime::PolicyController::new(case.mode, 64);
            let decision = controller.decide(signals);
            CaspObserved::PolicyAction(policy_action_name(&decision.action).to_owned())
        }
        CaspTestKind::SolverSelection => {
            let state_str = match &case.condition_state {
                Some(s) => s.as_str(),
                None => {
                    return CaspObserved::Error(
                        "solver_selection requires condition_state".to_owned(),
                    );
                }
            };
            let state = match parse_condition_state(state_str) {
                Some(s) => s,
                None => {
                    return CaspObserved::Error(format!("unknown condition state: {state_str}"));
                }
            };
            let portfolio = fsci_runtime::SolverPortfolio::new(case.mode, 64);
            let (action, _posterior, _losses, _loss) = portfolio.select_action(&state);
            CaspObserved::SolverAction(solver_action_name(&action).to_owned())
        }
        CaspTestKind::CalibratorDrift => {
            let observations = match &case.observations {
                Some(obs) => obs.clone(),
                None => {
                    return CaspObserved::Error(
                        "calibrator_drift requires observations".to_owned(),
                    );
                }
            };
            let mut calibrator = fsci_runtime::ConformalCalibrator::new(0.05, 200);
            for obs in &observations {
                calibrator.observe(*obs);
            }
            CaspObserved::CalibratorFallback(calibrator.should_fallback())
        }
    }
}

fn compare_casp_outcome(expected: &CaspExpectedOutcome, observed: &CaspObserved) -> (bool, String) {
    match (expected, observed) {
        (CaspExpectedOutcome::PolicyAction { action }, CaspObserved::PolicyAction(got)) => {
            let pass = action == got;
            if pass {
                (true, format!("policy action matched: {got}"))
            } else {
                (
                    false,
                    format!("policy action mismatch: expected '{action}', got '{got}'"),
                )
            }
        }
        (CaspExpectedOutcome::SolverAction { action }, CaspObserved::SolverAction(got)) => {
            let pass = action == got;
            if pass {
                (true, format!("solver action matched: {got}"))
            } else {
                (
                    false,
                    format!("solver action mismatch: expected '{action}', got '{got}'"),
                )
            }
        }
        (
            CaspExpectedOutcome::CalibratorFallback { should_fallback },
            CaspObserved::CalibratorFallback(got),
        ) => {
            let pass = *should_fallback == *got;
            if pass {
                (true, format!("calibrator fallback matched: {got}"))
            } else {
                (
                    false,
                    format!("calibrator fallback mismatch: expected {should_fallback}, got {got}"),
                )
            }
        }
        (CaspExpectedOutcome::Error { error }, CaspObserved::Error(got)) => {
            let pass = got.contains(error.as_str());
            if pass {
                (true, format!("error matched: {got}"))
            } else {
                (
                    false,
                    format!("error mismatch: expected '{error}', got '{got}'"),
                )
            }
        }
        (CaspExpectedOutcome::Error { error }, _) => {
            (false, format!("expected error '{error}' but got success"))
        }
        (_, CaspObserved::Error(e)) => (false, format!("unexpected error: {e}")),
        _ => (false, "outcome type mismatch".to_owned()),
    }
}

pub fn run_linalg_packet_with_oracle_capture(
    config: &HarnessConfig,
    fixture_name: &str,
    oracle: &PythonOracleConfig,
) -> Result<PacketReport, HarnessError> {
    let report = run_linalg_packet(config, fixture_name)?;
    let oracle_result = capture_linalg_oracle(config, fixture_name, oracle);
    if let Err(err) = oracle_result {
        if oracle.required {
            return Err(err);
        }
        let failure_path = config
            .artifact_dir_for(&report.packet_id)
            .join("oracle_capture.error.txt");
        fs::create_dir_all(config.artifact_dir_for(&report.packet_id)).map_err(|source| {
            HarnessError::ArtifactIo {
                path: config.artifact_dir_for(&report.packet_id),
                source,
            }
        })?;
        fs::write(&failure_path, format!("{err}")).map_err(|source| HarnessError::ArtifactIo {
            path: failure_path,
            source,
        })?;
    }

    Ok(report)
}

pub fn capture_linalg_oracle(
    config: &HarnessConfig,
    fixture_name: &str,
    oracle: &PythonOracleConfig,
) -> Result<PathBuf, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: LinalgPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.clone(),
            source,
        })?;

    let output_dir = config.artifact_dir_for(&fixture.packet_id);
    fs::create_dir_all(&output_dir).map_err(|source| HarnessError::ArtifactIo {
        path: output_dir.clone(),
        source,
    })?;
    let output_path = output_dir.join("oracle_capture.json");

    if !oracle.script_path.exists() {
        return Err(HarnessError::PythonScriptMissing {
            path: oracle.script_path.clone(),
        });
    }

    let python_bin = oracle.python_bin.display().to_string();
    let output = Command::new(&oracle.python_bin)
        .arg(&oracle.script_path)
        .arg(as_os_str("--fixture"))
        .arg(&fixture_path)
        .arg(as_os_str("--output"))
        .arg(&output_path)
        .arg(as_os_str("--oracle-root"))
        .arg(&config.oracle_root)
        .output()
        .map_err(|source| HarnessError::PythonLaunch {
            python_bin: python_bin.clone(),
            source,
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        if stderr.contains("No module named 'scipy'") {
            return Err(HarnessError::PythonSciPyMissing { stderr });
        }
        return Err(HarnessError::PythonFailed { python_bin, stderr });
    }

    let parsed = load_oracle_capture(&output_path)?;
    let normalized =
        serde_json::to_vec_pretty(&parsed).map_err(|e| HarnessError::RaptorQ(e.to_string()))?;
    fs::write(&output_path, normalized).map_err(|source| HarnessError::ArtifactIo {
        path: output_path.clone(),
        source,
    })?;

    Ok(output_path)
}

pub fn load_oracle_capture(path: &Path) -> Result<OracleCapture, HarnessError> {
    let raw = fs::read_to_string(path).map_err(|source| HarnessError::FixtureIo {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&raw).map_err(|source| HarnessError::OracleParse {
        path: path.to_path_buf(),
        source,
    })
}

pub fn load_packet_reports(config: &HarnessConfig) -> Result<Vec<PacketReport>, HarnessError> {
    let artifact_root = config.fixture_root.join("artifacts");
    if !artifact_root.exists() {
        return Ok(Vec::new());
    }

    let mut reports = Vec::new();
    for packet_dir in fs::read_dir(&artifact_root).map_err(|source| HarnessError::ArtifactIo {
        path: artifact_root.clone(),
        source,
    })? {
        let packet_dir = packet_dir
            .map_err(|source| HarnessError::ArtifactIo {
                path: artifact_root.clone(),
                source,
            })?
            .path();
        if !packet_dir.is_dir() {
            continue;
        }
        let report_path = packet_dir.join("parity_report.json");
        if !report_path.exists() {
            continue;
        }
        let raw = fs::read_to_string(&report_path).map_err(|source| HarnessError::ArtifactIo {
            path: report_path.clone(),
            source,
        })?;
        let report: PacketReport =
            serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
                path: report_path,
                source,
            })?;
        reports.push(report);
    }

    reports.sort_by(|a, b| a.packet_id.cmp(&b.packet_id));
    Ok(reports)
}

#[must_use]
pub fn packet_summary(report: &PacketReport) -> PacketSummary {
    PacketSummary {
        packet_id: report.packet_id.clone(),
        family: report.family.clone(),
        passed_cases: report.passed_cases,
        failed_cases: report.failed_cases,
        total_cases: report.case_results.len(),
    }
}

pub fn write_parity_artifacts(
    config: &HarnessConfig,
    report: &PacketReport,
) -> Result<ParityArtifactBundle, HarnessError> {
    let output_dir = config.artifact_dir_for(&report.packet_id);
    fs::create_dir_all(&output_dir).map_err(|source| HarnessError::ArtifactIo {
        path: output_dir.clone(),
        source,
    })?;

    let report_path = output_dir.join("parity_report.json");
    let report_bytes =
        serde_json::to_vec_pretty(report).map_err(|e| HarnessError::RaptorQ(e.to_string()))?;
    fs::write(&report_path, &report_bytes).map_err(|source| HarnessError::ArtifactIo {
        path: report_path.clone(),
        source,
    })?;

    let sidecar = generate_raptorq_sidecar(&report_bytes)?;
    let sidecar_path = output_dir.join("parity_report.raptorq.json");
    let sidecar_bytes =
        serde_json::to_vec_pretty(&sidecar).map_err(|e| HarnessError::RaptorQ(e.to_string()))?;
    fs::write(&sidecar_path, sidecar_bytes).map_err(|source| HarnessError::ArtifactIo {
        path: sidecar_path.clone(),
        source,
    })?;

    let decode_proof = DecodeProofArtifact {
        ts_unix_ms: now_unix_ms(),
        reason: "no recovery required; artifact remained intact".to_owned(),
        recovered_blocks: 0,
        proof_hash: hash(&report_bytes).to_hex().to_string(),
    };
    let decode_proof_path = output_dir.join("parity_report.decode_proof.json");
    let decode_proof_bytes = serde_json::to_vec_pretty(&decode_proof)
        .map_err(|e| HarnessError::RaptorQ(e.to_string()))?;
    fs::write(&decode_proof_path, decode_proof_bytes).map_err(|source| {
        HarnessError::ArtifactIo {
            path: decode_proof_path.clone(),
            source,
        }
    })?;

    Ok(ParityArtifactBundle {
        report_path,
        sidecar_path,
        decode_proof_path,
    })
}

#[cfg(feature = "dashboard")]
#[must_use]
pub fn style_for_case_result(case: &CaseResult) -> Style {
    if case.passed {
        Style::new().fg(PackedRgba::rgb(48, 186, 95)).bold()
    } else {
        Style::new().fg(PackedRgba::rgb(216, 63, 63)).bold()
    }
}

#[cfg(feature = "dashboard")]
#[must_use]
pub fn style_for_packet_summary(summary: &PacketSummary) -> Style {
    if summary.failed_cases == 0 {
        Style::new().fg(PackedRgba::rgb(70, 200, 120)).bold()
    } else {
        Style::new().fg(PackedRgba::rgb(220, 90, 70)).bold()
    }
}

fn execute_linalg_case(case: &LinalgCase) -> Result<LinalgObservedOutcome, LinalgError> {
    match case {
        LinalgCase::Solve {
            mode,
            a,
            b,
            assume_a,
            lower,
            transposed,
            check_finite,
            ..
        } => {
            let result = solve(
                a,
                b,
                SolveOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    assume_a: assume_a.clone().map(Into::into),
                    lower: lower.unwrap_or(false),
                    transposed: transposed.unwrap_or(false),
                },
            )?;
            Ok(LinalgObservedOutcome::Vector {
                values: result.x,
                warning_ill_conditioned: result.warning.is_some(),
            })
        }
        LinalgCase::SolveTriangular {
            mode,
            a,
            b,
            trans,
            lower,
            unit_diagonal,
            check_finite,
            ..
        } => {
            let result = solve_triangular(
                a,
                b,
                TriangularSolveOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    trans: trans
                        .clone()
                        .unwrap_or(FixtureTriangularTranspose::NoTranspose)
                        .into(),
                    lower: lower.unwrap_or(false),
                    unit_diagonal: unit_diagonal.unwrap_or(false),
                },
            )?;
            Ok(LinalgObservedOutcome::Vector {
                values: result.x,
                warning_ill_conditioned: false,
            })
        }
        LinalgCase::SolveBanded {
            mode,
            l_and_u,
            ab,
            b,
            check_finite,
            ..
        } => {
            let result = solve_banded(
                (l_and_u[0], l_and_u[1]),
                ab,
                b,
                SolveOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    assume_a: Some(MatrixAssumption::General),
                    lower: false,
                    transposed: false,
                },
            )?;
            Ok(LinalgObservedOutcome::Vector {
                values: result.x,
                warning_ill_conditioned: result.warning.is_some(),
            })
        }
        LinalgCase::Inv {
            mode,
            a,
            assume_a,
            lower,
            check_finite,
            ..
        } => {
            let result = inv(
                a,
                InvOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    assume_a: assume_a.clone().map(Into::into),
                    lower: lower.unwrap_or(false),
                },
            )?;
            Ok(LinalgObservedOutcome::Matrix {
                values: result.inverse,
            })
        }
        LinalgCase::Det {
            mode,
            a,
            check_finite,
            ..
        } => {
            let value = det(a, *mode, check_finite.unwrap_or(true))?;
            Ok(LinalgObservedOutcome::Scalar { value })
        }
        LinalgCase::Lstsq {
            mode,
            a,
            b,
            cond,
            check_finite,
            ..
        } => {
            let result = lstsq(
                a,
                b,
                LstsqOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    cond: *cond,
                    driver: LstsqDriver::default(),
                },
            )?;
            Ok(LinalgObservedOutcome::Lstsq {
                x: result.x,
                residuals: result.residuals,
                rank: result.rank,
                singular_values: result.singular_values,
            })
        }
        LinalgCase::Pinv {
            mode,
            a,
            atol,
            rtol,
            check_finite,
            ..
        } => {
            let result = pinv(
                a,
                PinvOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    atol: *atol,
                    rtol: *rtol,
                },
            )?;
            Ok(LinalgObservedOutcome::Pinv {
                values: result.pseudo_inverse,
                rank: result.rank,
            })
        }
    }
}

fn execute_optimize_case(case: &OptimizeCase) -> Result<OptimizeObservedOutcome, OptError> {
    match case {
        OptimizeCase::Minimize {
            mode,
            method,
            objective,
            x0,
            tol,
            maxiter,
            maxfev,
            ..
        } => {
            let objective = objective.clone();
            let result = minimize(
                |x| evaluate_minimize_objective(&objective, x),
                x0,
                MinimizeOptions {
                    method: Some(*method),
                    tol: *tol,
                    maxiter: *maxiter,
                    maxfev: *maxfev,
                    mode: *mode,
                    ..MinimizeOptions::default()
                },
            )?;
            Ok(OptimizeObservedOutcome::Minimize {
                x: result.x,
                fun: result.fun,
                success: result.success,
                status: result.status,
            })
        }
        OptimizeCase::Root {
            mode,
            method,
            objective,
            bracket,
            xtol,
            rtol,
            maxiter,
            ..
        } => {
            let objective = objective.clone();
            let options = RootOptions {
                method: Some(*method),
                xtol: xtol.unwrap_or(RootOptions::default().xtol),
                rtol: rtol.unwrap_or(RootOptions::default().rtol),
                maxiter: maxiter.unwrap_or(RootOptions::default().maxiter),
                mode: *mode,
                ..RootOptions::default()
            };
            let result = root_scalar(
                |x| evaluate_root_objective(&objective, x),
                Some((bracket[0], bracket[1])),
                None,
                None,
                options,
            )?;
            Ok(OptimizeObservedOutcome::Root {
                root: result.root,
                converged: result.converged,
                status: result.status,
            })
        }
    }
}

fn evaluate_minimize_objective(objective: &OptimizeMinObjective, x: &[f64]) -> f64 {
    let x0 = x.first().copied().unwrap_or(0.0);
    let x1 = x.get(1).copied().unwrap_or(0.0);
    match objective {
        OptimizeMinObjective::Rosenbrock2 => {
            let a = 1.0 - x0;
            let b = x1 - x0 * x0;
            a * a + 100.0 * b * b
        }
        OptimizeMinObjective::Ackley2 => {
            let s1 = 0.5 * (x0 * x0 + x1 * x1);
            let s2 = 0.5
                * ((2.0 * std::f64::consts::PI * x0).cos()
                    + (2.0 * std::f64::consts::PI * x1).cos());
            -20.0 * (-0.2 * s1.sqrt()).exp() - s2.exp() + std::f64::consts::E + 20.0
        }
        OptimizeMinObjective::Rastrigin2 => {
            20.0 + (x0 * x0 - 10.0 * (2.0 * std::f64::consts::PI * x0).cos())
                + (x1 * x1 - 10.0 * (2.0 * std::f64::consts::PI * x1).cos())
        }
        OptimizeMinObjective::Sphere2 => x0 * x0 + x1 * x1,
        OptimizeMinObjective::ShiftedQuadratic => {
            let dx = x0 - 1.0;
            let dy = x1 + 2.0;
            dx * dx + 4.0 * dy * dy
        }
        OptimizeMinObjective::TranslatedQuadratic => {
            let dx = x0 - 4.0;
            let dy = x1 + 3.0;
            dx * dx + 2.0 * dy * dy
        }
        OptimizeMinObjective::ScaledQuadratic => {
            10.0 * evaluate_minimize_objective(&OptimizeMinObjective::ShiftedQuadratic, x)
        }
        OptimizeMinObjective::RotatedQuadratic => {
            let cx = x0 - 1.0;
            let cy = x1 + 2.0;
            let theta = std::f64::consts::FRAC_PI_6;
            let xr = theta.cos() * cx - theta.sin() * cy;
            let yr = theta.sin() * cx + theta.cos() * cy;
            xr * xr + 4.0 * yr * yr
        }
        OptimizeMinObjective::L1Nonsmooth => x0.abs() + x1.abs(),
        OptimizeMinObjective::FlatQuartic => 1.0e-4 * (x0.powi(4) + x1.powi(4)),
        OptimizeMinObjective::NanBranch => {
            if x0 < 0.0 {
                f64::NAN
            } else {
                x0 * x0 + x1 * x1
            }
        }
    }
}

fn evaluate_root_objective(objective: &OptimizeRootObjective, x: f64) -> f64 {
    match objective {
        OptimizeRootObjective::CubicMinusTwo => x * x * x - 2.0,
        OptimizeRootObjective::CosMinusX => x.cos() - x,
        OptimizeRootObjective::SinMinusHalf => x.sin() - 0.5,
        OptimizeRootObjective::LinearShift03 => x - 0.3,
        OptimizeRootObjective::NanBranch => {
            if x > 0.8 {
                f64::NAN
            } else {
                x - 0.2
            }
        }
        OptimizeRootObjective::StepDiscontinuous => {
            if x < 0.3 {
                -1.0
            } else {
                1.0
            }
        }
    }
}

fn compare_optimize_case_differential(
    expected: &OptimizeExpectedOutcome,
    observed: &Result<OptimizeObservedOutcome, OptError>,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    match (expected, observed) {
        (
            OptimizeExpectedOutcome::MinimizePoint {
                x,
                fun,
                atol,
                rtol,
                require_success,
                contract_ref,
            },
            Ok(OptimizeObservedOutcome::Minimize {
                x: got_x,
                fun: got_fun,
                success,
                ..
            }),
        ) => {
            let tolerance = resolve_contract_tolerance(contract_ref.as_deref(), *atol, *rtol);
            let Some(got_fun) = *got_fun else {
                return (
                    false,
                    "minimize output missing objective value".to_owned(),
                    None,
                    Some(tolerance),
                );
            };
            let x_diff = if got_x.len() == x.len() {
                max_diff_vec(got_x, x)
            } else {
                f64::INFINITY
            };
            let fun_diff = (got_fun - *fun).abs();
            let pass = allclose_vec(got_x, x, tolerance.atol, tolerance.rtol)
                && allclose_scalar(got_fun, *fun, tolerance.atol, tolerance.rtol)
                && (!require_success.unwrap_or(true) || *success);
            let msg = if pass {
                format!(
                    "minimize matched (x_diff={x_diff:.2e}, fun_diff={fun_diff:.2e}, success={success})"
                )
            } else {
                format!(
                    "minimize mismatch: expected_x={x:?}, got_x={got_x:?}, expected_fun={fun:.8e}, got_fun={got_fun:.8e}, success={success}"
                )
            };
            (pass, msg, Some(x_diff.max(fun_diff)), Some(tolerance))
        }
        (
            OptimizeExpectedOutcome::RootValue {
                root,
                atol,
                rtol,
                require_converged,
                contract_ref,
            },
            Ok(OptimizeObservedOutcome::Root {
                root: got,
                converged,
                ..
            }),
        ) => {
            let tolerance = resolve_contract_tolerance(contract_ref.as_deref(), *atol, *rtol);
            let diff = (*got - *root).abs();
            let pass = allclose_scalar(*got, *root, tolerance.atol, tolerance.rtol)
                && (!require_converged.unwrap_or(true) || *converged);
            let msg = if pass {
                format!("root matched (diff={diff:.2e}, converged={converged})")
            } else {
                format!("root mismatch: expected={root:.8e}, got={got:.8e}, converged={converged}")
            };
            (pass, msg, Some(diff), Some(tolerance))
        }
        (
            OptimizeExpectedOutcome::MinimizeStatus { status, success },
            Ok(OptimizeObservedOutcome::Minimize {
                status: got_status,
                success: got_success,
                ..
            }),
        ) => {
            let pass = *status == *got_status && *success == *got_success;
            let msg = if pass {
                "minimize status matched".to_owned()
            } else {
                format!(
                    "minimize status mismatch: expected status={status:?}, success={success}; got status={got_status:?}, success={got_success}"
                )
            };
            (pass, msg, None, None)
        }
        (
            OptimizeExpectedOutcome::RootStatus { status, converged },
            Ok(OptimizeObservedOutcome::Root {
                status: got_status,
                converged: got_converged,
                ..
            }),
        ) => {
            let pass = *status == *got_status && *converged == *got_converged;
            let msg = if pass {
                "root status matched".to_owned()
            } else {
                format!(
                    "root status mismatch: expected status={status:?}, converged={converged}; got status={got_status:?}, converged={got_converged}"
                )
            };
            (pass, msg, None, None)
        }
        (OptimizeExpectedOutcome::Error { error }, Err(actual)) => {
            let pass = error == &actual.to_string();
            let msg = if pass {
                "error matched".to_owned()
            } else {
                format!("error mismatch: expected=`{error}`, got=`{actual}`")
            };
            (pass, msg, None, None)
        }
        (expected, observed) => (
            false,
            format!("shape mismatch: expected {expected:?}, got {observed:?}"),
            None,
            None,
        ),
    }
}

fn compare_linalg_case(
    expected: &LinalgExpectedOutcome,
    observed: &Result<LinalgObservedOutcome, LinalgError>,
) -> (bool, String) {
    match (expected, observed) {
        (
            LinalgExpectedOutcome::Vector {
                values,
                atol,
                rtol,
                expect_warning_ill_conditioned,
            },
            Ok(LinalgObservedOutcome::Vector {
                values: got,
                warning_ill_conditioned,
            }),
        ) => {
            let vectors_match = allclose_vec(got, values, *atol, *rtol);
            let warning_match = expect_warning_ill_conditioned
                .is_none_or(|expect| expect == *warning_ill_conditioned);
            let pass = vectors_match && warning_match;
            let msg = if pass {
                "linalg vector output matched expected tolerance policy".to_owned()
            } else {
                format!(
                    "vector mismatch: expected={values:?}, got={got:?}, atol={atol}, rtol={rtol}, expected_warning={expect_warning_ill_conditioned:?}, actual_warning={warning_ill_conditioned}"
                )
            };
            (pass, msg)
        }
        (
            LinalgExpectedOutcome::Matrix { values, atol, rtol },
            Ok(LinalgObservedOutcome::Matrix { values: got }),
        ) => {
            let pass = allclose_matrix(got, values, *atol, *rtol);
            let msg = if pass {
                "linalg matrix output matched expected tolerance policy".to_owned()
            } else {
                format!(
                    "matrix mismatch: expected={values:?}, got={got:?}, atol={atol}, rtol={rtol}"
                )
            };
            (pass, msg)
        }
        (
            LinalgExpectedOutcome::Scalar { value, atol, rtol },
            Ok(LinalgObservedOutcome::Scalar { value: got }),
        ) => {
            let pass = allclose_scalar(*got, *value, *atol, *rtol);
            let msg = if pass {
                "linalg scalar output matched expected tolerance policy".to_owned()
            } else {
                format!("scalar mismatch: expected={value}, got={got}, atol={atol}, rtol={rtol}")
            };
            (pass, msg)
        }
        (
            LinalgExpectedOutcome::Lstsq {
                x,
                residuals,
                rank,
                singular_values,
                atol,
                rtol,
            },
            Ok(LinalgObservedOutcome::Lstsq {
                x: got_x,
                residuals: got_res,
                rank: got_rank,
                singular_values: got_s,
            }),
        ) => {
            let pass = *rank == *got_rank
                && allclose_vec(got_x, x, *atol, *rtol)
                && allclose_vec(got_res, residuals, *atol, *rtol)
                && allclose_vec(got_s, singular_values, *atol, *rtol);
            let msg = if pass {
                "linalg lstsq output matched expected tolerance policy".to_owned()
            } else {
                format!(
                    "lstsq mismatch: expected_x={x:?}, got_x={got_x:?}, expected_res={residuals:?}, got_res={got_res:?}, expected_rank={rank}, got_rank={got_rank}, expected_s={singular_values:?}, got_s={got_s:?}"
                )
            };
            (pass, msg)
        }
        (
            LinalgExpectedOutcome::Pinv {
                values,
                rank,
                atol,
                rtol,
            },
            Ok(LinalgObservedOutcome::Pinv {
                values: got,
                rank: got_rank,
            }),
        ) => {
            let pass = *rank == *got_rank && allclose_matrix(got, values, *atol, *rtol);
            let msg = if pass {
                "linalg pinv output matched expected tolerance policy".to_owned()
            } else {
                format!(
                    "pinv mismatch: expected_values={values:?}, got_values={got:?}, expected_rank={rank}, got_rank={got_rank}, atol={atol}, rtol={rtol}"
                )
            };
            (pass, msg)
        }
        (LinalgExpectedOutcome::Error { error }, Err(actual)) => {
            let pass = error == &actual.to_string();
            let msg = if pass {
                "linalg error matched expected contract".to_owned()
            } else {
                format!("error mismatch: expected=`{error}`, got=`{actual}`")
            };
            (pass, msg)
        }
        (expected, got) => (false, format!("expected {expected:?} but observed {got:?}")),
    }
}

fn allclose_scalar(actual: f64, expected: f64, atol: f64, rtol: f64) -> bool {
    if actual.is_nan() && expected.is_nan() {
        return true;
    }
    (actual - expected).abs() <= atol + rtol * expected.abs()
}

fn allclose_vec(actual: &[f64], expected: &[f64], atol: f64, rtol: f64) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    actual
        .iter()
        .zip(expected.iter())
        .all(|(a, e)| allclose_scalar(*a, *e, atol, rtol))
}

fn allclose_matrix(actual: &[Vec<f64>], expected: &[Vec<f64>], atol: f64, rtol: f64) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    actual
        .iter()
        .zip(expected.iter())
        .all(|(a_row, e_row)| allclose_vec(a_row, e_row, atol, rtol))
}

fn build_packet_report(
    packet_id: String,
    family: String,
    case_results: Vec<CaseResult>,
) -> PacketReport {
    let passed_cases = case_results.iter().filter(|r| r.passed).count();
    let failed_cases = case_results.len().saturating_sub(passed_cases);
    PacketReport {
        packet_id,
        family,
        case_results,
        passed_cases,
        failed_cases,
        generated_unix_ms: now_unix_ms(),
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ContractTolerancePolicy {
    comparison_mode: Option<String>,
    default_atol: Option<f64>,
    default_rtol: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ContractEntry {
    function_name: String,
    tolerance_policy: ContractTolerancePolicy,
}

#[derive(Debug, Clone, Deserialize)]
struct ContractTable {
    contracts: Vec<ContractEntry>,
}

static OPTIMIZE_CONTRACT_TABLE: OnceLock<Option<ContractTable>> = OnceLock::new();
static SPECIAL_CONTRACT_TABLE: OnceLock<Option<ContractTable>> = OnceLock::new();
static ARRAY_API_CONTRACT_TABLE: OnceLock<Option<ContractTable>> = OnceLock::new();

fn load_optimize_contract_table() -> Option<&'static ContractTable> {
    OPTIMIZE_CONTRACT_TABLE
        .get_or_init(|| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/artifacts/P2C-003/contracts/contract_table.json");
            let raw = fs::read_to_string(path).ok()?;
            serde_json::from_str::<ContractTable>(&raw).ok()
        })
        .as_ref()
}

fn load_special_contract_table() -> Option<&'static ContractTable> {
    SPECIAL_CONTRACT_TABLE
        .get_or_init(|| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/artifacts/P2C-006/contracts/contract_table.json");
            let raw = fs::read_to_string(path).ok()?;
            serde_json::from_str::<ContractTable>(&raw).ok()
        })
        .as_ref()
}

fn load_array_api_contract_table() -> Option<&'static ContractTable> {
    ARRAY_API_CONTRACT_TABLE
        .get_or_init(|| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/artifacts/P2C-007/contracts/contract_table.json");
            let raw = fs::read_to_string(path).ok()?;
            serde_json::from_str::<ContractTable>(&raw).ok()
        })
        .as_ref()
}

fn resolve_contract_tolerance(
    contract_ref: Option<&str>,
    explicit_atol: Option<f64>,
    explicit_rtol: Option<f64>,
) -> ToleranceUsed {
    if let (Some(atol), Some(rtol)) = (explicit_atol, explicit_rtol) {
        return ToleranceUsed {
            atol,
            rtol,
            comparison_mode: "mixed".to_owned(),
        };
    }

    if let Some(contract_ref) = contract_ref
        && let Some(table) = load_optimize_contract_table()
        && let Some(entry) = table
            .contracts
            .iter()
            .find(|candidate| candidate.function_name == contract_ref)
    {
        return ToleranceUsed {
            atol: explicit_atol
                .or(entry.tolerance_policy.default_atol)
                .unwrap_or(1.0e-9),
            rtol: explicit_rtol
                .or(entry.tolerance_policy.default_rtol)
                .unwrap_or(1.0e-6),
            comparison_mode: entry
                .tolerance_policy
                .comparison_mode
                .clone()
                .unwrap_or_else(|| "mixed".to_owned()),
        };
    }

    ToleranceUsed {
        atol: explicit_atol.unwrap_or(1.0e-9),
        rtol: explicit_rtol.unwrap_or(1.0e-6),
        comparison_mode: "mixed".to_owned(),
    }
}

fn resolve_special_contract_tolerance(
    contract_ref: Option<&str>,
    explicit_atol: Option<f64>,
    explicit_rtol: Option<f64>,
) -> ToleranceUsed {
    if let (Some(atol), Some(rtol)) = (explicit_atol, explicit_rtol) {
        return ToleranceUsed {
            atol,
            rtol,
            comparison_mode: "mixed".to_owned(),
        };
    }

    if let Some(contract_ref) = contract_ref
        && let Some(table) = load_special_contract_table()
        && let Some(entry) = table
            .contracts
            .iter()
            .find(|candidate| candidate.function_name == contract_ref)
    {
        return ToleranceUsed {
            atol: explicit_atol
                .or(entry.tolerance_policy.default_atol)
                .unwrap_or(1.0e-9),
            rtol: explicit_rtol
                .or(entry.tolerance_policy.default_rtol)
                .unwrap_or(1.0e-6),
            comparison_mode: entry
                .tolerance_policy
                .comparison_mode
                .clone()
                .unwrap_or_else(|| "mixed".to_owned()),
        };
    }

    ToleranceUsed {
        atol: explicit_atol.unwrap_or(1.0e-9),
        rtol: explicit_rtol.unwrap_or(1.0e-6),
        comparison_mode: "mixed".to_owned(),
    }
}

fn resolve_array_api_contract_tolerance(
    contract_ref: Option<&str>,
    explicit_atol: Option<f64>,
    explicit_rtol: Option<f64>,
) -> ToleranceUsed {
    if let (Some(atol), Some(rtol)) = (explicit_atol, explicit_rtol) {
        return ToleranceUsed {
            atol,
            rtol,
            comparison_mode: "mixed".to_owned(),
        };
    }

    if let Some(contract_ref) = contract_ref
        && let Some(table) = load_array_api_contract_table()
        && let Some(entry) = table
            .contracts
            .iter()
            .find(|candidate| candidate.function_name == contract_ref)
    {
        let comparison_mode = entry
            .tolerance_policy
            .comparison_mode
            .clone()
            .unwrap_or_else(|| "exact".to_owned());
        return ToleranceUsed {
            atol: explicit_atol
                .or(entry.tolerance_policy.default_atol)
                .unwrap_or(0.0),
            rtol: explicit_rtol
                .or(entry.tolerance_policy.default_rtol)
                .unwrap_or(0.0),
            comparison_mode,
        };
    }

    ToleranceUsed {
        atol: explicit_atol.unwrap_or(0.0),
        rtol: explicit_rtol.unwrap_or(0.0),
        comparison_mode: "exact".to_owned(),
    }
}

pub fn generate_raptorq_sidecar(payload: &[u8]) -> Result<RaptorQSidecar, HarnessError> {
    let symbol_size = 128usize;
    let source_symbols = chunk_payload(payload, symbol_size);
    let k = source_symbols.len();
    let repair_symbols = (k / 5).max(1);
    let seed = hash(payload).as_bytes()[0] as u64 + 1337;

    let encoder = SystematicEncoder::new(&source_symbols, symbol_size, seed).ok_or_else(|| {
        HarnessError::RaptorQ("systematic encoder initialization failed".to_owned())
    })?;

    let mut repair_hashes = Vec::with_capacity(repair_symbols);
    for esi in k as u32..(k as u32 + repair_symbols as u32) {
        let symbol = encoder.repair_symbol(esi);
        repair_hashes.push(hash(&symbol).to_hex().to_string());
    }

    Ok(RaptorQSidecar {
        schema_version: 1,
        source_hash: hash(payload).to_hex().to_string(),
        symbol_size,
        source_symbols: k,
        repair_symbols,
        repair_symbol_hashes: repair_hashes,
    })
}

pub fn chunk_payload(payload: &[u8], symbol_size: usize) -> Vec<Vec<u8>> {
    if payload.is_empty() {
        return vec![vec![0u8; symbol_size]];
    }

    let mut chunks = Vec::with_capacity(payload.len().div_ceil(symbol_size));
    for chunk in payload.chunks(symbol_size) {
        let mut symbol = vec![0u8; symbol_size];
        let len = chunk.len();
        symbol[..len].copy_from_slice(chunk);
        chunks.push(symbol);
    }
    chunks
}

fn as_os_str(input: &str) -> &OsStr {
    OsStr::new(input)
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

// ═══════════════════════════════════════════════════════════════════
// Differential Conformance Harness (bd-3jh.4)
// Generic API: run_differential_test(fixture_path, oracle_config) -> ConformanceReport
// ═══════════════════════════════════════════════════════════════════

/// Oracle configuration for differential conformance testing.
#[derive(Debug, Clone)]
pub struct DifferentialOracleConfig {
    /// Path to Python interpreter (e.g., ".venv-py314/bin/python3").
    pub python_path: PathBuf,
    /// Path to the oracle script that runs SciPy and emits JSON.
    pub script_path: PathBuf,
    /// Maximum time in seconds for oracle execution.
    pub timeout_secs: u64,
    /// If true, oracle failures are hard errors. If false, oracle_missing is acceptable.
    pub required: bool,
}

impl Default for DifferentialOracleConfig {
    fn default() -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            python_path: PathBuf::from("python3"),
            script_path: manifest.join("python_oracle/scipy_linalg_oracle.py"),
            timeout_secs: 30,
            required: false,
        }
    }
}

/// Status of the Python/SciPy oracle for a given test run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum OracleStatus {
    /// Oracle ran successfully and produced results.
    Available,
    /// Oracle was not available (SciPy not installed, script missing, etc.).
    Missing { reason: String },
    /// Oracle exceeded the configured timeout.
    TimedOut,
    /// Oracle ran but produced an error.
    Failed { reason: String },
    /// Oracle was not invoked (not configured or not needed).
    Skipped,
}

/// Tolerance values that were used for a particular case comparison.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToleranceUsed {
    pub atol: f64,
    pub rtol: f64,
    pub comparison_mode: String,
}

/// Per-case result from differential conformance testing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DifferentialCaseResult {
    /// Unique case identifier within the fixture.
    pub case_id: String,
    /// Whether the case passed the conformance check.
    pub passed: bool,
    /// Human-readable description of the result or failure.
    pub message: String,
    /// Maximum absolute difference between Rust and expected output.
    /// None if the case involves error matching rather than numeric comparison.
    pub max_diff: Option<f64>,
    /// The tolerance values used for this comparison.
    pub tolerance_used: Option<ToleranceUsed>,
    /// Status of the oracle for this particular case.
    pub oracle_status: OracleStatus,
}

/// Conformance report produced by `run_differential_test`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConformanceReport {
    /// Path to the fixture file that was tested.
    pub fixture_path: String,
    /// Packet identifier extracted from the fixture.
    pub packet_id: String,
    /// Family name from the fixture (e.g., "validate_tol", "linalg_core").
    pub family: String,
    /// Number of cases that passed.
    pub pass_count: usize,
    /// Number of cases that failed.
    pub fail_count: usize,
    /// Overall oracle status for the test run.
    pub oracle_status: OracleStatus,
    /// Per-case results with max_diff and tolerance information.
    pub per_case_results: Vec<DifferentialCaseResult>,
    /// Timestamp when this report was generated.
    pub generated_unix_ms: u128,
}

/// Fixture envelope: minimal structure to detect fixture type.
#[derive(Debug, Clone, Deserialize)]
struct FixtureEnvelope {
    family: String,
}

/// Run a differential conformance test against a fixture file.
///
/// This is the generic entry point for the conformance harness. It:
/// 1. Loads and parses the fixture file
/// 2. Detects the fixture type (validate_tol, linalg_core, etc.)
/// 3. Runs the Rust implementation against each case
/// 4. Optionally runs the Python/SciPy oracle for comparison
/// 5. Computes max_diff between Rust and expected outputs per case
///
/// If the oracle is unavailable and `oracle_config.required == false`,
/// the report marks oracle_status as `Missing` and still validates
/// against the fixture's embedded expected values.
pub fn run_differential_test(
    fixture_path: &Path,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let raw = fs::read_to_string(fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.to_path_buf(),
        source,
    })?;

    let envelope: FixtureEnvelope =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let family = envelope.family.as_str();
    if family.contains("validate_tol") {
        run_differential_validate_tol(fixture_path, &raw, oracle_config)
    } else if family.contains("linalg") {
        run_differential_linalg(fixture_path, &raw, oracle_config)
    } else if family.contains("array_api") {
        run_differential_array_api(fixture_path, &raw, oracle_config)
    } else if family.contains("special") {
        run_differential_special(fixture_path, &raw, oracle_config)
    } else if family.contains("optim") {
        run_differential_optimize(fixture_path, &raw, oracle_config)
    } else {
        Err(HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source: serde::de::Error::custom(format!("unknown fixture family: {family}")),
        })
    }
}

fn run_differential_validate_tol(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: PacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());

    for case in &fixture.cases {
        let outcome = validate_tol(
            case.rtol.clone().into(),
            case.atol.clone().into(),
            case.n,
            case.mode,
        );

        let (passed, message, max_diff, tolerance_used) = match (&case.expected, outcome) {
            (
                ExpectedOutcome::Ok {
                    rtol,
                    atol,
                    warning_rtol_clamped,
                },
                Ok(actual),
            ) => {
                let actual_warning = !actual.warnings.is_empty();
                let expected_rtol: ToleranceValue = rtol.clone().into();
                let expected_atol: ToleranceValue = atol.clone().into();
                let pass = actual.rtol == expected_rtol
                    && actual.atol == expected_atol
                    && actual_warning == *warning_rtol_clamped;
                let msg = if pass {
                    "tolerance contract matched".to_owned()
                } else {
                    format!(
                        "tolerance mismatch: expected rtol={expected_rtol:?}, atol={expected_atol:?}, warning={warning_rtol_clamped}; got rtol={:?}, atol={:?}, warning={actual_warning}",
                        actual.rtol, actual.atol
                    )
                };
                (pass, msg, None, None)
            }
            (ExpectedOutcome::Error { error }, Err(actual)) => {
                let pass = error == &actual.to_string();
                let msg = if pass {
                    "error matched".to_owned()
                } else {
                    format!("error mismatch: expected `{error}`, got `{actual}`")
                };
                (pass, msg, None, None)
            }
            (expected, result) => (
                false,
                format!("shape mismatch: expected {expected:?}, got {result:?}"),
                None,
                None,
            ),
        };

        per_case_results.push(DifferentialCaseResult {
            case_id: case.case_id.clone(),
            passed,
            message,
            max_diff,
            tolerance_used,
            oracle_status: oracle_status.clone(),
        });
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_linalg(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: LinalgPacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());

    for case in &fixture.cases {
        let observed = execute_linalg_case(case);
        let (passed, message, max_diff, tolerance_used) =
            compare_linalg_case_differential(case.expected(), &observed);

        per_case_results.push(DifferentialCaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
            max_diff,
            tolerance_used,
            oracle_status: oracle_status.clone(),
        });
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_array_api(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: ArrayApiPacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());

    for case in &fixture.cases {
        let observed = execute_array_api_case(case);
        let (passed, message, max_diff, tolerance_used) =
            compare_array_api_case_differential(case, &observed);
        per_case_results.push(DifferentialCaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
            max_diff,
            tolerance_used,
            oracle_status: oracle_status.clone(),
        });
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_optimize(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: OptimizePacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());

    for case in &fixture.cases {
        let observed = execute_optimize_case(case);
        let (passed, message, max_diff, tolerance_used) =
            compare_optimize_case_differential(case.expected(), &observed);

        per_case_results.push(DifferentialCaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
            max_diff,
            tolerance_used,
            oracle_status: oracle_status.clone(),
        });
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_special(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: SpecialPacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());

    for case in &fixture.cases {
        let observed = execute_special_case(case);
        let (passed, message, max_diff, tolerance_used) =
            compare_special_case_differential(case, &observed);

        per_case_results.push(DifferentialCaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
            max_diff,
            tolerance_used,
            oracle_status: oracle_status.clone(),
        });
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn execute_array_api_case(case: &ArrayApiCase) -> Result<ArrayApiObservedOutcome, String> {
    let mode = array_api_exec_mode(match case {
        ArrayApiCase::Zeros { mode, .. }
        | ArrayApiCase::Ones { mode, .. }
        | ArrayApiCase::Full { mode, .. }
        | ArrayApiCase::Arange { mode, .. }
        | ArrayApiCase::Linspace { mode, .. }
        | ArrayApiCase::BroadcastShapes { mode, .. }
        | ArrayApiCase::ResultType { mode, .. }
        | ArrayApiCase::FromSlice { mode, .. }
        | ArrayApiCase::Getitem { mode, .. }
        | ArrayApiCase::Reshape { mode, .. }
        | ArrayApiCase::Transpose { mode, .. }
        | ArrayApiCase::RelationBroadcastCommutative { mode, .. }
        | ArrayApiCase::RelationResultTypeSymmetry { mode, .. }
        | ArrayApiCase::RelationIndexRoundtrip { mode, .. } => *mode,
    });
    let backend = FsciArrayApiCoreArrayBackend::new(mode);

    match case {
        ArrayApiCase::Zeros {
            shape,
            dtype,
            expected: _,
            ..
        } => {
            let request = ArrayApiCreationRequest {
                shape: FsciArrayApiShape::new(shape.clone()),
                dtype: fixture_dtype_to_runtime(*dtype),
                order: FsciArrayApiMemoryOrder::C,
            };
            let array = arrayapi_zeros(&backend, &request).map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::Ones {
            shape,
            dtype,
            expected: _,
            ..
        } => {
            let request = ArrayApiCreationRequest {
                shape: FsciArrayApiShape::new(shape.clone()),
                dtype: fixture_dtype_to_runtime(*dtype),
                order: FsciArrayApiMemoryOrder::C,
            };
            let array = arrayapi_ones(&backend, &request).map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::Full {
            shape,
            fill_value,
            dtype,
            expected: _,
            ..
        } => {
            let request = ArrayApiFullRequest {
                fill_value: fixture_scalar_to_runtime(*fill_value),
                dtype: fixture_dtype_to_runtime(*dtype),
                order: FsciArrayApiMemoryOrder::C,
            };
            let array = arrayapi_full(&backend, &FsciArrayApiShape::new(shape.clone()), &request)
                .map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::Arange {
            start,
            stop,
            step,
            dtype,
            expected: _,
            ..
        } => {
            let request = ArrayApiArangeRequest {
                start: fixture_scalar_to_runtime(*start),
                stop: fixture_scalar_to_runtime(*stop),
                step: fixture_scalar_to_runtime(*step),
                dtype: dtype.map(fixture_dtype_to_runtime),
            };
            let array = arrayapi_arange(&backend, &request).map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::Linspace {
            start,
            stop,
            num,
            endpoint,
            dtype,
            expected: _,
            ..
        } => {
            let request = ArrayApiLinspaceRequest {
                start: fixture_scalar_to_runtime(*start),
                stop: fixture_scalar_to_runtime(*stop),
                num: *num,
                endpoint: *endpoint,
                dtype: dtype.map(fixture_dtype_to_runtime),
            };
            let array = arrayapi_linspace(&backend, &request).map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::BroadcastShapes {
            shapes,
            expected: _,
            ..
        } => {
            let runtime_shapes = shapes
                .iter()
                .map(|shape| FsciArrayApiShape::new(shape.clone()))
                .collect::<Vec<_>>();
            let out = arrayapi_broadcast_shapes(&runtime_shapes).map_err(array_api_error_kind)?;
            Ok(ArrayApiObservedOutcome::Shape { dims: out.dims })
        }
        ArrayApiCase::ResultType {
            dtypes,
            force_floating,
            expected: _,
            ..
        } => {
            let runtime_dtypes = dtypes
                .iter()
                .map(|dtype| fixture_dtype_to_runtime(*dtype))
                .collect::<Vec<_>>();
            let dtype = arrayapi_result_type(&backend, &runtime_dtypes, *force_floating)
                .map_err(array_api_error_kind)?;
            Ok(ArrayApiObservedOutcome::Dtype {
                dtype: runtime_dtype_to_fixture(dtype),
            })
        }
        ArrayApiCase::FromSlice {
            values,
            shape,
            dtype,
            expected: _,
            ..
        } => {
            let request = ArrayApiCreationRequest {
                shape: FsciArrayApiShape::new(shape.clone()),
                dtype: fixture_dtype_to_runtime(*dtype),
                order: FsciArrayApiMemoryOrder::C,
            };
            let runtime_values = values
                .iter()
                .map(|value| fixture_scalar_to_runtime(*value))
                .collect::<Vec<_>>();
            let array = arrayapi_from_slice(&backend, &runtime_values, &request)
                .map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::Getitem {
            source_values,
            source_shape,
            source_dtype,
            indexing_mode,
            index,
            expected: _,
            ..
        } => {
            let source = array_api_source_array(
                &backend,
                source_values,
                source_shape,
                *source_dtype,
                FsciArrayApiMemoryOrder::C,
            )?;
            let request = ArrayApiIndexRequest {
                mode: fixture_indexing_mode_to_runtime(*indexing_mode),
                index: fixture_index_to_runtime(index),
            };
            let array =
                arrayapi_getitem(&backend, &source, &request).map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::Reshape {
            source_values,
            source_shape,
            source_dtype,
            new_shape,
            expected: _,
            ..
        } => {
            let source = array_api_source_array(
                &backend,
                source_values,
                source_shape,
                *source_dtype,
                FsciArrayApiMemoryOrder::C,
            )?;
            let array = arrayapi_reshape(
                &backend,
                &source,
                &FsciArrayApiShape::new(new_shape.clone()),
            )
            .map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::Transpose {
            source_values,
            source_shape,
            source_dtype,
            expected: _,
            ..
        } => {
            let source = array_api_source_array(
                &backend,
                source_values,
                source_shape,
                *source_dtype,
                FsciArrayApiMemoryOrder::C,
            )?;
            let array = arrayapi_transpose(&backend, &source).map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::RelationBroadcastCommutative {
            left_shape,
            right_shape,
            expected: _,
            ..
        } => {
            let left = FsciArrayApiShape::new(left_shape.clone());
            let right = FsciArrayApiShape::new(right_shape.clone());
            let lr = arrayapi_broadcast_shapes(&[left.clone(), right.clone()]);
            let rl = arrayapi_broadcast_shapes(&[right, left]);
            let pass = match (lr, rl) {
                (Ok(lhs), Ok(rhs)) => lhs == rhs,
                (Err(_), Err(_)) => true,
                _ => false,
            };
            Ok(ArrayApiObservedOutcome::Bool { value: pass })
        }
        ArrayApiCase::RelationResultTypeSymmetry {
            left_dtype,
            right_dtype,
            force_floating,
            expected: _,
            ..
        } => {
            let left = fixture_dtype_to_runtime(*left_dtype);
            let right = fixture_dtype_to_runtime(*right_dtype);
            let forward = arrayapi_result_type(&backend, &[left, right], *force_floating);
            let reverse = arrayapi_result_type(&backend, &[right, left], *force_floating);
            let pass = match (forward, reverse) {
                (Ok(a), Ok(b)) => a == b,
                _ => false,
            };
            Ok(ArrayApiObservedOutcome::Bool { value: pass })
        }
        ArrayApiCase::RelationIndexRoundtrip {
            values,
            dtype,
            index,
            expected: _,
            ..
        } => {
            let len = values.len();
            let source = array_api_source_array(
                &backend,
                values,
                &[len],
                *dtype,
                FsciArrayApiMemoryOrder::C,
            )?;
            let request = ArrayApiIndexRequest {
                mode: FsciArrayApiIndexingMode::Advanced,
                index: FsciArrayApiIndexExpr::Advanced {
                    indices: vec![vec![*index]],
                },
            };
            let selected =
                arrayapi_getitem(&backend, &source, &request).map_err(array_api_error_kind)?;
            let normalized = if *index < 0 {
                (len as isize + *index) as usize
            } else {
                *index as usize
            };
            let expected_value = fixture_scalar_to_runtime(values[normalized]);
            let pass = selected.size() == 1
                && selected
                    .values()
                    .first()
                    .is_some_and(|value| *value == expected_value);
            Ok(ArrayApiObservedOutcome::Bool { value: pass })
        }
    }
}

fn compare_array_api_case_differential(
    case: &ArrayApiCase,
    observed: &Result<ArrayApiObservedOutcome, String>,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    match (case.expected(), observed) {
        (
            ArrayApiExpectedOutcome::Array {
                shape,
                dtype,
                values,
                atol,
                rtol,
                contract_ref,
            },
            Ok(ArrayApiObservedOutcome::Array {
                shape: got_shape,
                dtype: got_dtype,
                values: got_values,
            }),
        ) => {
            let tolerance =
                resolve_array_api_contract_tolerance(contract_ref.as_deref(), *atol, *rtol);
            let shape_pass = got_shape == shape;
            let dtype_pass = got_dtype == dtype;
            let values_pass = got_values.len() == values.len()
                && got_values
                    .iter()
                    .zip(values.iter())
                    .all(|(actual, expected)| {
                        fixture_scalar_allclose(actual, expected, tolerance.atol, tolerance.rtol)
                    });
            let pass = shape_pass && dtype_pass && values_pass;
            let max_diff = fixture_max_diff_seq(got_values, values);
            let msg = if pass {
                format!(
                    "array matched shape={shape:?} dtype={dtype:?} (max_diff={:.2e})",
                    max_diff
                )
            } else {
                format!(
                    "array mismatch: expected shape={shape:?} dtype={dtype:?}, got shape={got_shape:?} dtype={got_dtype:?}"
                )
            };
            (pass, msg, Some(max_diff), Some(tolerance))
        }
        (
            ArrayApiExpectedOutcome::Shape { dims, contract_ref },
            Ok(ArrayApiObservedOutcome::Shape { dims: got }),
        ) => {
            let pass = got == dims;
            let msg = if pass {
                format!("shape matched {dims:?}")
            } else {
                format!("shape mismatch: expected {dims:?}, got {got:?}")
            };
            let tolerance = contract_ref
                .as_deref()
                .map(|value| resolve_array_api_contract_tolerance(Some(value), None, None));
            (pass, msg, None, tolerance)
        }
        (
            ArrayApiExpectedOutcome::Dtype {
                dtype,
                contract_ref,
            },
            Ok(ArrayApiObservedOutcome::Dtype { dtype: got }),
        ) => {
            let pass = got == dtype;
            let msg = if pass {
                format!("dtype matched {dtype:?}")
            } else {
                format!("dtype mismatch: expected {dtype:?}, got {got:?}")
            };
            let tolerance = contract_ref
                .as_deref()
                .map(|value| resolve_array_api_contract_tolerance(Some(value), None, None));
            (pass, msg, None, tolerance)
        }
        (
            ArrayApiExpectedOutcome::Bool {
                value,
                contract_ref,
            },
            Ok(ArrayApiObservedOutcome::Bool { value: got }),
        ) => {
            let pass = got == value;
            let msg = if pass {
                format!("relation matched ({value})")
            } else {
                format!("relation mismatch: expected={value}, got={got}")
            };
            let tolerance = contract_ref
                .as_deref()
                .map(|value| resolve_array_api_contract_tolerance(Some(value), None, None));
            (pass, msg, None, tolerance)
        }
        (ArrayApiExpectedOutcome::ErrorKind { error }, Err(observed_kind)) => {
            let pass = observed_kind == error;
            let msg = if pass {
                format!("error kind matched ({observed_kind})")
            } else {
                format!("error kind mismatch: expected={error}, got={observed_kind}")
            };
            (pass, msg, None, None)
        }
        (expected, observed) => (
            false,
            format!("shape mismatch: expected {expected:?}, got {observed:?}"),
            None,
            None,
        ),
    }
}

fn array_api_source_array(
    backend: &FsciArrayApiCoreArrayBackend,
    values: &[ArrayApiFixtureScalar],
    shape: &[usize],
    dtype: ArrayApiFixtureDType,
    order: FsciArrayApiMemoryOrder,
) -> Result<FsciArrayApiCoreArray, String> {
    let request = ArrayApiCreationRequest {
        shape: FsciArrayApiShape::new(shape.to_vec()),
        dtype: fixture_dtype_to_runtime(dtype),
        order,
    };
    let runtime_values = values
        .iter()
        .map(|value| fixture_scalar_to_runtime(*value))
        .collect::<Vec<_>>();
    arrayapi_from_slice(backend, &runtime_values, &request).map_err(array_api_error_kind)
}

fn observed_array(array: &FsciArrayApiCoreArray) -> ArrayApiObservedOutcome {
    ArrayApiObservedOutcome::Array {
        shape: array.shape().dims.clone(),
        dtype: runtime_dtype_to_fixture(array.dtype()),
        values: array
            .values()
            .iter()
            .map(|value| runtime_scalar_to_fixture(*value))
            .collect(),
    }
}

fn array_api_exec_mode(mode: RuntimeMode) -> FsciArrayApiExecutionMode {
    match mode {
        RuntimeMode::Strict => FsciArrayApiExecutionMode::Strict,
        RuntimeMode::Hardened => FsciArrayApiExecutionMode::Hardened,
    }
}

fn fixture_dtype_to_runtime(dtype: ArrayApiFixtureDType) -> FsciArrayApiDType {
    match dtype {
        ArrayApiFixtureDType::Bool => FsciArrayApiDType::Bool,
        ArrayApiFixtureDType::Int64 => FsciArrayApiDType::Int64,
        ArrayApiFixtureDType::UInt64 => FsciArrayApiDType::UInt64,
        ArrayApiFixtureDType::Float32 => FsciArrayApiDType::Float32,
        ArrayApiFixtureDType::Float64 => FsciArrayApiDType::Float64,
        ArrayApiFixtureDType::Complex64 => FsciArrayApiDType::Complex64,
        ArrayApiFixtureDType::Complex128 => FsciArrayApiDType::Complex128,
    }
}

fn runtime_dtype_to_fixture(dtype: FsciArrayApiDType) -> ArrayApiFixtureDType {
    match dtype {
        FsciArrayApiDType::Bool => ArrayApiFixtureDType::Bool,
        FsciArrayApiDType::Int64 => ArrayApiFixtureDType::Int64,
        FsciArrayApiDType::UInt64 => ArrayApiFixtureDType::UInt64,
        FsciArrayApiDType::Float32 => ArrayApiFixtureDType::Float32,
        FsciArrayApiDType::Float64 => ArrayApiFixtureDType::Float64,
        FsciArrayApiDType::Complex64 => ArrayApiFixtureDType::Complex64,
        FsciArrayApiDType::Complex128 => ArrayApiFixtureDType::Complex128,
        FsciArrayApiDType::Int8
        | FsciArrayApiDType::Int16
        | FsciArrayApiDType::Int32
        | FsciArrayApiDType::UInt8
        | FsciArrayApiDType::UInt16
        | FsciArrayApiDType::UInt32 => ArrayApiFixtureDType::Int64,
    }
}

fn fixture_scalar_to_runtime(value: ArrayApiFixtureScalar) -> FsciArrayApiScalarValue {
    match value {
        ArrayApiFixtureScalar::Bool { value } => FsciArrayApiScalarValue::Bool(value),
        ArrayApiFixtureScalar::I64 { value } => FsciArrayApiScalarValue::I64(value),
        ArrayApiFixtureScalar::U64 { value } => FsciArrayApiScalarValue::U64(value),
        ArrayApiFixtureScalar::F64 { value } => FsciArrayApiScalarValue::F64(value),
        ArrayApiFixtureScalar::ComplexF64 { re, im } => {
            FsciArrayApiScalarValue::ComplexF64 { re, im }
        }
    }
}

fn runtime_scalar_to_fixture(value: FsciArrayApiScalarValue) -> ArrayApiFixtureScalar {
    match value {
        FsciArrayApiScalarValue::Bool(value) => ArrayApiFixtureScalar::Bool { value },
        FsciArrayApiScalarValue::I64(value) => ArrayApiFixtureScalar::I64 { value },
        FsciArrayApiScalarValue::U64(value) => ArrayApiFixtureScalar::U64 { value },
        FsciArrayApiScalarValue::F64(value) => ArrayApiFixtureScalar::F64 { value },
        FsciArrayApiScalarValue::ComplexF64 { re, im } => {
            ArrayApiFixtureScalar::ComplexF64 { re, im }
        }
    }
}

fn fixture_indexing_mode_to_runtime(mode: ArrayApiFixtureIndexingMode) -> FsciArrayApiIndexingMode {
    match mode {
        ArrayApiFixtureIndexingMode::Basic => FsciArrayApiIndexingMode::Basic,
        ArrayApiFixtureIndexingMode::Advanced => FsciArrayApiIndexingMode::Advanced,
        ArrayApiFixtureIndexingMode::BooleanMask => FsciArrayApiIndexingMode::BooleanMask,
    }
}

fn fixture_index_to_runtime(index: &ArrayApiFixtureIndexExpr) -> FsciArrayApiIndexExpr {
    match index {
        ArrayApiFixtureIndexExpr::Basic { slices } => FsciArrayApiIndexExpr::Basic {
            slices: slices
                .iter()
                .map(|slice| FsciArrayApiSliceSpec {
                    start: slice.start,
                    stop: slice.stop,
                    step: slice.step,
                })
                .collect(),
        },
        ArrayApiFixtureIndexExpr::Advanced { indices } => FsciArrayApiIndexExpr::Advanced {
            indices: indices.clone(),
        },
        ArrayApiFixtureIndexExpr::BooleanMask { mask_shape } => {
            FsciArrayApiIndexExpr::BooleanMask {
                mask_shape: FsciArrayApiShape::new(mask_shape.clone()),
            }
        }
    }
}

fn array_api_error_kind(error: fsci_arrayapi::ArrayApiError) -> String {
    format!("{:?}", error.kind)
}

fn fixture_scalar_diff(actual: &ArrayApiFixtureScalar, expected: &ArrayApiFixtureScalar) -> f64 {
    match (actual, expected) {
        (ArrayApiFixtureScalar::Bool { value: a }, ArrayApiFixtureScalar::Bool { value: e }) => {
            if a == e { 0.0 } else { 1.0 }
        }
        (ArrayApiFixtureScalar::I64 { value: a }, ArrayApiFixtureScalar::I64 { value: e }) => {
            (*a as f64 - *e as f64).abs()
        }
        (ArrayApiFixtureScalar::U64 { value: a }, ArrayApiFixtureScalar::U64 { value: e }) => {
            (*a as f64 - *e as f64).abs()
        }
        (ArrayApiFixtureScalar::F64 { value: a }, ArrayApiFixtureScalar::F64 { value: e }) => {
            (*a - *e).abs()
        }
        (
            ArrayApiFixtureScalar::ComplexF64 { re: ar, im: ai },
            ArrayApiFixtureScalar::ComplexF64 { re: er, im: ei },
        ) => (ar - er).abs().max((ai - ei).abs()),
        _ => f64::INFINITY,
    }
}

fn fixture_scalar_allclose(
    actual: &ArrayApiFixtureScalar,
    expected: &ArrayApiFixtureScalar,
    atol: f64,
    rtol: f64,
) -> bool {
    match (actual, expected) {
        (ArrayApiFixtureScalar::Bool { value: a }, ArrayApiFixtureScalar::Bool { value: e }) => {
            a == e
        }
        (ArrayApiFixtureScalar::I64 { value: a }, ArrayApiFixtureScalar::I64 { value: e }) => {
            a == e
        }
        (ArrayApiFixtureScalar::U64 { value: a }, ArrayApiFixtureScalar::U64 { value: e }) => {
            a == e
        }
        (ArrayApiFixtureScalar::F64 { value: a }, ArrayApiFixtureScalar::F64 { value: e }) => {
            allclose_scalar(*a, *e, atol, rtol)
        }
        (
            ArrayApiFixtureScalar::ComplexF64 { re: ar, im: ai },
            ArrayApiFixtureScalar::ComplexF64 { re: er, im: ei },
        ) => allclose_scalar(*ar, *er, atol, rtol) && allclose_scalar(*ai, *ei, atol, rtol),
        _ => false,
    }
}

fn fixture_max_diff_seq(
    actual: &[ArrayApiFixtureScalar],
    expected: &[ArrayApiFixtureScalar],
) -> f64 {
    if actual.len() != expected.len() {
        return f64::INFINITY;
    }
    actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| fixture_scalar_diff(a, e))
        .fold(0.0_f64, f64::max)
}

fn execute_special_case(case: &SpecialCase) -> Result<f64, FsciSpecialError> {
    let mode = case.mode;
    let args = &case.args;
    match case.function {
        SpecialCaseFunction::Gamma => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("gamma", mode));
            }
            special_scalar_from_tensor(
                special_gamma(&special_scalar(args[0]), mode)?,
                "gamma",
                mode,
            )
        }
        SpecialCaseFunction::Gammaln => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("gammaln", mode));
            }
            special_scalar_from_tensor(
                special_gammaln(&special_scalar(args[0]), mode)?,
                "gammaln",
                mode,
            )
        }
        SpecialCaseFunction::Gammainc => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("gammainc", mode));
            }
            special_scalar_from_tensor(
                special_gammainc(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "gammainc",
                mode,
            )
        }
        SpecialCaseFunction::Gammaincc => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("gammaincc", mode));
            }
            special_scalar_from_tensor(
                special_gammaincc(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "gammaincc",
                mode,
            )
        }
        SpecialCaseFunction::Erf => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("erf", mode));
            }
            special_scalar_from_tensor(special_erf(&special_scalar(args[0]), mode)?, "erf", mode)
        }
        SpecialCaseFunction::Erfc => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("erfc", mode));
            }
            special_scalar_from_tensor(special_erfc(&special_scalar(args[0]), mode)?, "erfc", mode)
        }
        SpecialCaseFunction::Erfinv => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("erfinv", mode));
            }
            special_scalar_from_tensor(
                special_erfinv(&special_scalar(args[0]), mode)?,
                "erfinv",
                mode,
            )
        }
        SpecialCaseFunction::Erfcinv => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("erfcinv", mode));
            }
            special_scalar_from_tensor(
                special_erfcinv(&special_scalar(args[0]), mode)?,
                "erfcinv",
                mode,
            )
        }
        SpecialCaseFunction::Beta => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("beta", mode));
            }
            special_scalar_from_tensor(
                special_beta(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "beta",
                mode,
            )
        }
        SpecialCaseFunction::Betaln => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("betaln", mode));
            }
            special_scalar_from_tensor(
                special_betaln(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "betaln",
                mode,
            )
        }
        SpecialCaseFunction::Betainc => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("betainc", mode));
            }
            special_scalar_from_tensor(
                special_betainc(
                    &special_scalar(args[0]),
                    &special_scalar(args[1]),
                    &special_scalar(args[2]),
                    mode,
                )?,
                "betainc",
                mode,
            )
        }
        SpecialCaseFunction::J0 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("j0", mode));
            }
            special_scalar_from_tensor(special_j0(&special_scalar(args[0]), mode)?, "j0", mode)
        }
        SpecialCaseFunction::J1 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("j1", mode));
            }
            special_scalar_from_tensor(special_j1(&special_scalar(args[0]), mode)?, "j1", mode)
        }
        SpecialCaseFunction::Jn => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("jn", mode));
            }
            special_scalar_from_tensor(
                special_jn(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "jn",
                mode,
            )
        }
        SpecialCaseFunction::Y0 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("y0", mode));
            }
            special_scalar_from_tensor(special_y0(&special_scalar(args[0]), mode)?, "y0", mode)
        }
        SpecialCaseFunction::Y1 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("y1", mode));
            }
            special_scalar_from_tensor(special_y1(&special_scalar(args[0]), mode)?, "y1", mode)
        }
        SpecialCaseFunction::Yn => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("yn", mode));
            }
            special_scalar_from_tensor(
                special_yn(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "yn",
                mode,
            )
        }
        SpecialCaseFunction::RelErfErfcIdentity => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("rel_erf_erfc_identity", mode));
            }
            let x = args[0];
            let erf_v =
                special_scalar_from_tensor(special_erf(&special_scalar(x), mode)?, "erf", mode)?;
            let erfc_v =
                special_scalar_from_tensor(special_erfc(&special_scalar(x), mode)?, "erfc", mode)?;
            Ok(erf_v + erfc_v)
        }
        SpecialCaseFunction::RelGammaRecurrence => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("rel_gamma_recurrence", mode));
            }
            let x = args[0];
            let gx = special_scalar_from_tensor(
                special_gamma(&special_scalar(x), mode)?,
                "gamma",
                mode,
            )?;
            let gx1 = special_scalar_from_tensor(
                special_gamma(&special_scalar(x + 1.0), mode)?,
                "gamma",
                mode,
            )?;
            Ok(gx1 - x * gx)
        }
        SpecialCaseFunction::RelBetaSymmetry => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("rel_beta_symmetry", mode));
            }
            let a = args[0];
            let b = args[1];
            let lhs = special_scalar_from_tensor(
                special_beta(&special_scalar(a), &special_scalar(b), mode)?,
                "beta",
                mode,
            )?;
            let rhs = special_scalar_from_tensor(
                special_beta(&special_scalar(b), &special_scalar(a), mode)?,
                "beta",
                mode,
            )?;
            Ok(lhs - rhs)
        }
        SpecialCaseFunction::RelGammaincComplement => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error(
                    "rel_gammainc_complement",
                    mode,
                ));
            }
            let a = args[0];
            let x = args[1];
            let p = special_scalar_from_tensor(
                special_gammainc(&special_scalar(a), &special_scalar(x), mode)?,
                "gammainc",
                mode,
            )?;
            let q = special_scalar_from_tensor(
                special_gammaincc(&special_scalar(a), &special_scalar(x), mode)?,
                "gammaincc",
                mode,
            )?;
            Ok(p + q)
        }
        SpecialCaseFunction::RelJnRecurrence => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("rel_jn_recurrence", mode));
            }
            let n = args[0];
            let x = args[1];
            let jn_prev = special_scalar_from_tensor(
                special_jn(&special_scalar(n - 1.0), &special_scalar(x), mode)?,
                "jn",
                mode,
            )?;
            let jn_curr = special_scalar_from_tensor(
                special_jn(&special_scalar(n), &special_scalar(x), mode)?,
                "jn",
                mode,
            )?;
            let jn_next = special_scalar_from_tensor(
                special_jn(&special_scalar(n + 1.0), &special_scalar(x), mode)?,
                "jn",
                mode,
            )?;
            Ok(jn_next - ((2.0 * n / x) * jn_curr - jn_prev))
        }
        SpecialCaseFunction::RelErfinvComposition => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error(
                    "rel_erfinv_composition",
                    mode,
                ));
            }
            let x = args[0];
            let erf_x =
                special_scalar_from_tensor(special_erf(&special_scalar(x), mode)?, "erf", mode)?;
            special_scalar_from_tensor(
                special_erfinv(&special_scalar(erf_x), mode)?,
                "erfinv",
                mode,
            )
        }
    }
}

fn special_scalar(value: f64) -> FsciSpecialTensor {
    FsciSpecialTensor::RealScalar(value)
}

fn special_scalar_from_tensor(
    tensor: FsciSpecialTensor,
    function: &'static str,
    mode: RuntimeMode,
) -> Result<f64, FsciSpecialError> {
    match tensor {
        FsciSpecialTensor::RealScalar(value) => Ok(value),
        _ => Err(FsciSpecialError {
            function,
            kind: FsciSpecialErrorKind::NotYetImplemented,
            mode,
            detail: "expected scalar output in conformance fixture",
        }),
    }
}

fn special_invalid_fixture_error(function: &'static str, mode: RuntimeMode) -> FsciSpecialError {
    FsciSpecialError {
        function,
        kind: FsciSpecialErrorKind::DomainError,
        mode,
        detail: "invalid fixture arity for special case",
    }
}

fn compare_special_case_differential(
    case: &SpecialCase,
    observed: &Result<f64, FsciSpecialError>,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    match (&case.expected, observed) {
        (
            SpecialExpectedOutcome::Scalar {
                value,
                atol,
                rtol,
                contract_ref,
            },
            Ok(actual),
        ) => {
            let tolerance =
                resolve_special_contract_tolerance(contract_ref.as_deref(), *atol, *rtol);
            let diff = (*actual - *value).abs();
            let pass = allclose_scalar(*actual, *value, tolerance.atol, tolerance.rtol);
            let msg = if pass {
                format!("special scalar matched (diff={diff:.2e})")
            } else {
                format!(
                    "special scalar mismatch: expected={value:.16e}, got={actual:.16e}, diff={diff:.2e}, atol={:.2e}, rtol={:.2e}",
                    tolerance.atol, tolerance.rtol
                )
            };
            (pass, msg, Some(diff), Some(tolerance))
        }
        (SpecialExpectedOutcome::Class { class }, Ok(actual)) => {
            let observed_class = classify_special_value(*actual);
            let pass = observed_class == *class;
            let msg = if pass {
                format!("value class matched ({observed_class:?})")
            } else {
                format!("value class mismatch: expected={class:?}, got={observed_class:?}")
            };
            (pass, msg, None, None)
        }
        (SpecialExpectedOutcome::ErrorKind { error }, Err(actual)) => {
            let observed_kind = format!("{:?}", actual.kind);
            let pass = observed_kind == *error;
            let msg = if pass {
                format!("error kind matched ({observed_kind})")
            } else {
                format!("error kind mismatch: expected={error}, got={observed_kind}")
            };
            (pass, msg, None, None)
        }
        (expected, observed) => (
            false,
            format!("shape mismatch: expected {expected:?}, got {observed:?}"),
            None,
            None,
        ),
    }
}

fn classify_special_value(value: f64) -> SpecialValueClass {
    if value.is_nan() {
        return SpecialValueClass::Nan;
    }
    if value == f64::INFINITY {
        return SpecialValueClass::PosInf;
    }
    if value == f64::NEG_INFINITY {
        return SpecialValueClass::NegInf;
    }
    SpecialValueClass::Finite
}

/// Compare a linalg case and return (passed, message, max_diff, tolerance_used).
fn compare_linalg_case_differential(
    expected: &LinalgExpectedOutcome,
    observed: &Result<LinalgObservedOutcome, LinalgError>,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    match (expected, observed) {
        (
            LinalgExpectedOutcome::Vector {
                values,
                atol,
                rtol,
                expect_warning_ill_conditioned,
            },
            Ok(LinalgObservedOutcome::Vector {
                values: got,
                warning_ill_conditioned,
            }),
        ) => {
            let md = max_diff_vec(got, values);
            let vectors_match = allclose_vec(got, values, *atol, *rtol);
            let warning_match = expect_warning_ill_conditioned
                .is_none_or(|expect| expect == *warning_ill_conditioned);
            let pass = vectors_match && warning_match;
            let msg = if pass {
                format!("vector matched (max_diff={md:.2e})")
            } else {
                format!("vector mismatch: max_diff={md:.2e}, atol={atol}, rtol={rtol}")
            };
            (
                pass,
                msg,
                Some(md),
                Some(ToleranceUsed {
                    atol: *atol,
                    rtol: *rtol,
                    comparison_mode: "mixed".to_owned(),
                }),
            )
        }
        (
            LinalgExpectedOutcome::Matrix { values, atol, rtol },
            Ok(LinalgObservedOutcome::Matrix { values: got }),
        ) => {
            let md = max_diff_matrix(got, values);
            let pass = allclose_matrix(got, values, *atol, *rtol);
            let msg = if pass {
                format!("matrix matched (max_diff={md:.2e})")
            } else {
                format!("matrix mismatch: max_diff={md:.2e}, atol={atol}, rtol={rtol}")
            };
            (
                pass,
                msg,
                Some(md),
                Some(ToleranceUsed {
                    atol: *atol,
                    rtol: *rtol,
                    comparison_mode: "mixed".to_owned(),
                }),
            )
        }
        (
            LinalgExpectedOutcome::Scalar { value, atol, rtol },
            Ok(LinalgObservedOutcome::Scalar { value: got }),
        ) => {
            let md = (*got - *value).abs();
            let pass = allclose_scalar(*got, *value, *atol, *rtol);
            let msg = if pass {
                format!("scalar matched (diff={md:.2e})")
            } else {
                format!("scalar mismatch: diff={md:.2e}, atol={atol}, rtol={rtol}")
            };
            (
                pass,
                msg,
                Some(md),
                Some(ToleranceUsed {
                    atol: *atol,
                    rtol: *rtol,
                    comparison_mode: "mixed".to_owned(),
                }),
            )
        }
        (
            LinalgExpectedOutcome::Lstsq {
                x,
                residuals,
                rank,
                singular_values,
                atol,
                rtol,
            },
            Ok(LinalgObservedOutcome::Lstsq {
                x: got_x,
                residuals: got_res,
                rank: got_rank,
                singular_values: got_s,
            }),
        ) => {
            let md = max_diff_vec(got_x, x)
                .max(max_diff_vec(got_res, residuals))
                .max(max_diff_vec(got_s, singular_values));
            let pass = *rank == *got_rank
                && allclose_vec(got_x, x, *atol, *rtol)
                && allclose_vec(got_res, residuals, *atol, *rtol)
                && allclose_vec(got_s, singular_values, *atol, *rtol);
            let msg = if pass {
                format!("lstsq matched (max_diff={md:.2e})")
            } else {
                format!("lstsq mismatch: max_diff={md:.2e}, rank expected={rank} got={got_rank}")
            };
            (
                pass,
                msg,
                Some(md),
                Some(ToleranceUsed {
                    atol: *atol,
                    rtol: *rtol,
                    comparison_mode: "mixed".to_owned(),
                }),
            )
        }
        (
            LinalgExpectedOutcome::Pinv {
                values,
                rank,
                atol,
                rtol,
            },
            Ok(LinalgObservedOutcome::Pinv {
                values: got,
                rank: got_rank,
            }),
        ) => {
            let md = max_diff_matrix(got, values);
            let pass = *rank == *got_rank && allclose_matrix(got, values, *atol, *rtol);
            let msg = if pass {
                format!("pinv matched (max_diff={md:.2e})")
            } else {
                format!("pinv mismatch: max_diff={md:.2e}, rank expected={rank} got={got_rank}")
            };
            (
                pass,
                msg,
                Some(md),
                Some(ToleranceUsed {
                    atol: *atol,
                    rtol: *rtol,
                    comparison_mode: "mixed".to_owned(),
                }),
            )
        }
        (LinalgExpectedOutcome::Error { error }, Err(actual)) => {
            let pass = error == &actual.to_string();
            let msg = if pass {
                "error matched".to_owned()
            } else {
                format!("error mismatch: expected=`{error}`, got=`{actual}`")
            };
            (pass, msg, None, None)
        }
        (expected, got) => (
            false,
            format!("shape mismatch: expected {expected:?} but observed {got:?}"),
            None,
            None,
        ),
    }
}

/// Compute the maximum absolute difference between two vectors.
fn max_diff_vec(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Compute the maximum absolute difference between two matrices.
fn max_diff_matrix(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ar, br)| max_diff_vec(ar, br))
        .fold(0.0_f64, f64::max)
}

/// Probe whether the oracle is available without running a full test.
fn probe_oracle_availability(config: &DifferentialOracleConfig) -> OracleStatus {
    if !config.script_path.exists() {
        return OracleStatus::Missing {
            reason: format!("script not found: {}", config.script_path.display()),
        };
    }

    let result = Command::new(&config.python_path)
        .arg("-c")
        .arg("import scipy; print(scipy.__version__)")
        .output();

    match result {
        Ok(output) if output.status.success() => OracleStatus::Available,
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            if stderr.contains("No module named") {
                OracleStatus::Missing {
                    reason: format!("scipy not available: {stderr}"),
                }
            } else {
                OracleStatus::Failed { reason: stderr }
            }
        }
        Err(e) => OracleStatus::Missing {
            reason: format!("python not found: {e}"),
        },
    }
}

// ═══════════════════════════════════════════════════════════════════
// Packet Runner Registry (§bd-3jh.19.10)
// ═══════════════════════════════════════════════════════════════════

/// Known packet families across the FrankenSciPy conformance surface.
///
/// Each variant maps to one P2C ticket and its associated fixture format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PacketFamily {
    /// P2C-001: Integration tolerance validation
    ValidateTol,
    /// P2C-002: Dense linear algebra (solve, inv, det, lstsq, pinv)
    LinalgCore,
    /// P2C-003: Optimization and root-finding routines
    Optimize,
    /// P2C-004: Sparse matrix operations
    SparseOps,
    /// P2C-005: FFT routines
    Fft,
    /// P2C-006: Special functions
    Special,
    /// P2C-007: Array API compatibility
    ArrayApi,
    /// P2C-008: CASP runtime (PolicyController, SolverPortfolio, ConformalCalibrator)
    RuntimeCasp,
}

impl PacketFamily {
    /// All known packet families for enumeration.
    pub const ALL: [Self; 8] = [
        Self::ValidateTol,
        Self::LinalgCore,
        Self::Optimize,
        Self::SparseOps,
        Self::Fft,
        Self::Special,
        Self::ArrayApi,
        Self::RuntimeCasp,
    ];

    /// Canonical packet ID for this family (e.g., "FSCI-P2C-001").
    #[must_use]
    pub const fn packet_id(&self) -> &'static str {
        match self {
            Self::ValidateTol => "FSCI-P2C-001",
            Self::LinalgCore => "FSCI-P2C-002",
            Self::Optimize => "FSCI-P2C-003",
            Self::SparseOps => "FSCI-P2C-004",
            Self::Fft => "FSCI-P2C-005",
            Self::Special => "FSCI-P2C-006",
            Self::ArrayApi => "FSCI-P2C-007",
            Self::RuntimeCasp => "FSCI-P2C-008",
        }
    }

    /// Short name used in fixture file naming (e.g., "validate_tol").
    #[must_use]
    pub const fn family_name(&self) -> &'static str {
        match self {
            Self::ValidateTol => "validate_tol",
            Self::LinalgCore => "linalg_core",
            Self::Optimize => "optimize",
            Self::SparseOps => "sparse_ops",
            Self::Fft => "fft",
            Self::Special => "special",
            Self::ArrayApi => "array_api",
            Self::RuntimeCasp => "runtime_casp",
        }
    }

    /// Detect family from a fixture family string.
    #[must_use]
    pub fn from_family_str(s: &str) -> Option<Self> {
        if s.contains("validate_tol") {
            Some(Self::ValidateTol)
        } else if s.contains("linalg") {
            Some(Self::LinalgCore)
        } else if s.contains("sparse") {
            Some(Self::SparseOps)
        } else if s.contains("fft") {
            Some(Self::Fft)
        } else if s.contains("special") {
            Some(Self::Special)
        } else if s.contains("optim") {
            Some(Self::Optimize)
        } else if s.contains("array_api") {
            Some(Self::ArrayApi)
        } else if s.contains("runtime") || s.contains("casp") {
            Some(Self::RuntimeCasp)
        } else {
            None
        }
    }

    /// Whether this family has a working runner implementation.
    #[must_use]
    pub const fn has_runner(&self) -> bool {
        matches!(
            self,
            Self::ValidateTol
                | Self::LinalgCore
                | Self::Optimize
                | Self::SparseOps
                | Self::Fft
                | Self::Special
                | Self::ArrayApi
                | Self::RuntimeCasp
        )
    }

    /// Canonical fixture filename for this family.
    #[must_use]
    pub fn fixture_filename(&self) -> String {
        format!("{}_{}.json", self.packet_id(), self.family_name())
    }
}

/// Aggregate parity report merging results from multiple packet families.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateParityReport {
    /// Timestamp of aggregation.
    pub generated_unix_ms: u128,
    /// Per-packet summaries.
    pub packets: Vec<PacketSummary>,
    /// Total cases across all packets.
    pub total_cases: usize,
    /// Total passed across all packets.
    pub total_passed: usize,
    /// Total failed across all packets.
    pub total_failed: usize,
}

/// Aggregate individual packet reports into a cross-packet parity report.
#[must_use]
pub fn aggregate_packet_reports(reports: &[PacketReport]) -> AggregateParityReport {
    let packets: Vec<PacketSummary> = reports.iter().map(packet_summary).collect();
    let total_cases: usize = packets.iter().map(|p| p.total_cases).sum();
    let total_passed: usize = packets.iter().map(|p| p.passed_cases).sum();
    let total_failed: usize = packets.iter().map(|p| p.failed_cases).sum();

    AggregateParityReport {
        generated_unix_ms: now_unix_ms(),
        packets,
        total_cases,
        total_passed,
        total_failed,
    }
}

/// Ensure the canonical artifact directory layout exists for a packet.
///
/// Creates: `{fixture_root}/artifacts/{packet_id}/{subdir}` for each
/// required subdirectory (anchor, contracts, threats).
pub fn ensure_artifact_layout(
    config: &HarnessConfig,
    packet_id: &str,
) -> Result<PathBuf, HarnessError> {
    let base = config.artifact_dir_for(packet_id);
    for subdir in &["anchor", "contracts", "threats"] {
        let dir = base.join(subdir);
        if !dir.exists() {
            fs::create_dir_all(&dir)
                .map_err(|source| HarnessError::ArtifactIo { path: dir, source })?;
        }
    }
    Ok(base)
}

/// Discover all fixture files in the fixture root and return their paths
/// grouped by detected packet family.
pub fn discover_fixtures(
    config: &HarnessConfig,
) -> Result<Vec<(PacketFamily, PathBuf)>, HarnessError> {
    let mut results = Vec::new();
    let entries = fs::read_dir(&config.fixture_root).map_err(|source| HarnessError::FixtureIo {
        path: config.fixture_root.clone(),
        source,
    })?;

    for entry in entries {
        let entry = entry.map_err(|source| HarnessError::FixtureIo {
            path: config.fixture_root.clone(),
            source,
        })?;
        let path = entry.path();
        if path.extension().and_then(OsStr::to_str) != Some("json") {
            continue;
        }
        let name = path.file_stem().and_then(OsStr::to_str).unwrap_or_default();
        // Skip non-fixture files (smoke_case, etc.)
        if !name.starts_with("FSCI-P2C-") {
            continue;
        }
        // Try to detect family from filename
        if let Some(family) = PacketFamily::from_family_str(name) {
            results.push((family, path));
        }
    }

    results.sort_by(|a, b| a.0.packet_id().cmp(b.0.packet_id()));
    Ok(results)
}

/// Run all discoverable fixtures and produce an aggregate report.
pub fn run_all_packets(config: &HarnessConfig) -> Result<AggregateParityReport, HarnessError> {
    let fixtures = discover_fixtures(config)?;
    let mut reports = Vec::new();

    for (family, path) in &fixtures {
        if !family.has_runner() {
            continue;
        }
        let fixture_name = path.file_name().and_then(OsStr::to_str).unwrap_or_default();
        let report = match family {
            PacketFamily::ValidateTol => run_validate_tol_packet(config, fixture_name)?,
            PacketFamily::LinalgCore => run_linalg_packet(config, fixture_name)?,
            PacketFamily::Optimize => run_optimize_packet(config, fixture_name)?,
            PacketFamily::SparseOps => run_sparse_packet(config, fixture_name)?,
            PacketFamily::Fft => run_fft_packet(config, fixture_name)?,
            PacketFamily::Special => run_special_packet(config, fixture_name)?,
            PacketFamily::ArrayApi => run_array_api_packet(config, fixture_name)?,
            PacketFamily::RuntimeCasp => run_casp_packet(config, fixture_name)?,
        };
        reports.push(report);
    }

    Ok(aggregate_packet_reports(&reports))
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "dashboard")]
    use super::style_for_case_result;
    use super::{
        AggregateParityReport, ArrayApiPacketFixture, ConformanceReport, DifferentialOracleConfig,
        HarnessConfig, LinalgPacketFixture, OptimizePacketFixture, OracleStatus, PacketFamily,
        PythonOracleConfig, SpecialPacketFixture, aggregate_packet_reports, discover_fixtures,
        ensure_artifact_layout, load_oracle_capture, run_array_api_packet, run_casp_packet,
        run_differential_test, run_fft_packet, run_linalg_packet, run_optimize_packet, run_smoke,
        run_sparse_packet, run_special_packet, run_validate_tol_packet, write_parity_artifacts,
    };
    use serde::Serialize;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    #[test]
    fn smoke_harness_finds_oracle_and_fixtures() {
        let cfg = HarnessConfig::default_paths();
        let report = run_smoke(&cfg);
        assert!(report.oracle_present, "oracle repo should be present");
        assert!(report.fixture_count >= 1, "expected at least one fixture");
        assert!(report.strict_mode);
    }

    #[test]
    fn validate_tol_packet_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_validate_tol_packet(&cfg, "FSCI-P2C-001_validate_tol.json")
            .expect("packet fixture should run");
        assert_eq!(report.failed_cases, 0);
        assert!(report.passed_cases >= 1);

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("artifact generation must pass");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());

        #[cfg(feature = "dashboard")]
        {
            let styled = style_for_case_result(report.case_results.first().expect("case exists"));
            assert!(
                styled.fg.is_some(),
                "style must be colorized for dashboard rendering"
            );
        }
    }

    #[test]
    fn linalg_packet_passes() {
        let cfg = HarnessConfig::default_paths();
        let report =
            run_linalg_packet(&cfg, "FSCI-P2C-002_linalg_core.json").expect("linalg packet runs");
        assert_eq!(report.failed_cases, 0);
        assert!(report.passed_cases >= 1);

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("linalg parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn mock_python_oracle_capture_parses() {
        let unique = format!("fsci-conformance-test-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let fixtures = root.join("fixtures");
        fs::create_dir_all(&fixtures).expect("create fixtures");

        let fixture_src = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-002_linalg_core.json");
        let fixture_dst = fixtures.join("FSCI-P2C-002_linalg_core.json");
        fs::copy(&fixture_src, &fixture_dst).expect("copy fixture");

        let script_path = root.join("mock_oracle.py");
        let script = r#"
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--oracle-root", required=True)
args = parser.parse_args()

fixture = json.loads(Path(args.fixture).read_text())
result = {
    "packet_id": fixture["packet_id"],
    "family": fixture["family"],
    "generated_unix_ms": 0,
    "case_outputs": [
        {
            "case_id": c["case_id"],
            "status": "ok",
            "result_kind": "mock",
            "result": {"oracle": "mock"},
            "error": None,
        }
        for c in fixture["cases"]
    ],
}
Path(args.output).write_text(json.dumps(result, indent=2))
"#;
        fs::write(&script_path, script).expect("write script");

        let cfg = HarnessConfig {
            oracle_root: PathBuf::from("/tmp/nonexistent-oracle"),
            fixture_root: fixtures,
            strict_mode: true,
        };
        let oracle = PythonOracleConfig {
            python_bin: PathBuf::from("python3"),
            script_path,
            required: true,
        };

        let output_path =
            super::capture_linalg_oracle(&cfg, "FSCI-P2C-002_linalg_core.json", &oracle)
                .expect("mock oracle capture succeeds");
        let parsed = load_oracle_capture(&output_path).expect("oracle capture parse succeeds");
        assert_eq!(parsed.packet_id, "FSCI-P2C-002");
        assert!(!parsed.case_outputs.is_empty());

        let fixture_raw = fs::read_to_string(fixture_dst).expect("read fixture");
        let fixture: LinalgPacketFixture =
            serde_json::from_str(&fixture_raw).expect("fixture parse");
        assert_eq!(parsed.case_outputs.len(), fixture.cases.len());
    }

    #[test]
    fn scipy_oracle_capture_when_available() {
        let scipy_check = Command::new("python3")
            .arg("-c")
            .arg("import scipy")
            .status();
        if !matches!(scipy_check, Ok(status) if status.success()) {
            eprintln!("SciPy not available in this environment; skipping optional oracle test");
            return;
        }

        let unique = format!("fsci-conformance-scipy-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let fixtures = root.join("fixtures");
        fs::create_dir_all(&fixtures).expect("create fixtures");

        let fixture_src = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-002_linalg_core.json");
        let fixture_dst = fixtures.join("FSCI-P2C-002_linalg_core.json");
        fs::copy(&fixture_src, &fixture_dst).expect("copy fixture");

        let cfg = HarnessConfig {
            oracle_root: HarnessConfig::default_paths().oracle_root,
            fixture_root: fixtures,
            strict_mode: true,
        };

        let oracle = PythonOracleConfig {
            required: true,
            ..PythonOracleConfig::default()
        };

        let output_path =
            super::capture_linalg_oracle(&cfg, "FSCI-P2C-002_linalg_core.json", &oracle)
                .expect("scipy oracle capture succeeds");
        let parsed = load_oracle_capture(&output_path).expect("oracle capture parse succeeds");

        let fixture_raw = fs::read_to_string(fixture_dst).expect("read fixture");
        let fixture: LinalgPacketFixture =
            serde_json::from_str(&fixture_raw).expect("fixture parse");
        assert_eq!(parsed.packet_id, "FSCI-P2C-002");
        assert_eq!(parsed.case_outputs.len(), fixture.cases.len());
    }

    // ═══════════════════════════════════════════════════════════════
    // Differential conformance harness tests (bd-3jh.4)
    // ═══════════════════════════════════════════════════════════════

    fn default_test_oracle() -> DifferentialOracleConfig {
        DifferentialOracleConfig {
            required: false,
            ..DifferentialOracleConfig::default()
        }
    }

    #[derive(Debug, Serialize)]
    struct StructuredCaseLog {
        test_id: String,
        category: String,
        input_summary: String,
        expected: String,
        actual: String,
        diff: Option<f64>,
        tolerance: Option<super::ToleranceUsed>,
        pass: bool,
    }

    fn optimize_case_input_summary(case: &super::OptimizeCase) -> String {
        match case {
            super::OptimizeCase::Minimize {
                method,
                objective,
                x0,
                ..
            } => format!("operation=minimize method={method:?} objective={objective:?} x0={x0:?}"),
            super::OptimizeCase::Root {
                method,
                objective,
                bracket,
                ..
            } => format!(
                "operation=root method={method:?} objective={objective:?} bracket={bracket:?}"
            ),
        }
    }

    fn optimize_case_expected_summary(expected: &super::OptimizeExpectedOutcome) -> String {
        match expected {
            super::OptimizeExpectedOutcome::MinimizePoint { x, fun, .. } => {
                format!("minimize_point x={x:?} fun={fun:.6e}")
            }
            super::OptimizeExpectedOutcome::RootValue { root, .. } => {
                format!("root_value root={root:.6e}")
            }
            super::OptimizeExpectedOutcome::MinimizeStatus { status, success } => {
                format!("minimize_status status={status:?} success={success}")
            }
            super::OptimizeExpectedOutcome::RootStatus { status, converged } => {
                format!("root_status status={status:?} converged={converged}")
            }
            super::OptimizeExpectedOutcome::Error { error } => format!("error={error}"),
        }
    }

    fn special_case_input_summary(case: &super::SpecialCase) -> String {
        format!(
            "function={:?} mode={:?} args={:?}",
            case.function, case.mode, case.args
        )
    }

    fn special_case_expected_summary(expected: &super::SpecialExpectedOutcome) -> String {
        match expected {
            super::SpecialExpectedOutcome::Scalar {
                value,
                atol,
                rtol,
                contract_ref,
            } => format!(
                "scalar value={value:.16e} atol={atol:?} rtol={rtol:?} contract_ref={contract_ref:?}"
            ),
            super::SpecialExpectedOutcome::Class { class } => {
                format!("class={class:?}")
            }
            super::SpecialExpectedOutcome::ErrorKind { error } => {
                format!("error_kind={error}")
            }
        }
    }

    fn array_api_case_input_summary(case: &super::ArrayApiCase) -> String {
        format!("{case:?}")
    }

    fn array_api_case_expected_summary(expected: &super::ArrayApiExpectedOutcome) -> String {
        format!("{expected:?}")
    }

    #[test]
    fn differential_test_validate_tol_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-001_validate_tol.json");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle)
            .expect("differential validate_tol should succeed");

        assert_eq!(report.packet_id, "FSCI-P2C-001");
        assert_eq!(report.family, "integrate.validate_tol");
        assert_eq!(report.fail_count, 0);
        assert!(report.pass_count >= 1);
        assert_eq!(
            report.per_case_results.len(),
            report.pass_count + report.fail_count
        );
    }

    #[test]
    fn differential_test_linalg_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-002_linalg_core.json");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle)
            .expect("differential linalg should succeed");

        assert_eq!(report.packet_id, "FSCI-P2C-002");
        assert_eq!(report.family, "linalg_core");
        assert_eq!(report.fail_count, 0);
        assert!(report.pass_count >= 1);

        // Linalg cases should produce max_diff values
        for case in &report.per_case_results {
            assert!(
                case.passed,
                "case {} should pass: {}",
                case.case_id, case.message
            );
            // Numeric cases have max_diff; error cases don't
            if case.tolerance_used.is_some() {
                assert!(
                    case.max_diff.is_some(),
                    "numeric case {} should have max_diff",
                    case.case_id
                );
            }
        }
    }

    #[test]
    fn optimize_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_optimize_packet(&cfg, "FSCI-P2C-003_optimize_core.json")
            .expect("optimize packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 29,
            "expected full optimize fixture execution"
        );

        let artifacts = write_parity_artifacts(&cfg, &report)
            .expect("optimize parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn special_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_special_packet(&cfg, "FSCI-P2C-006_special_core.json")
            .expect("special packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one special function test case"
        );

        let artifacts = write_parity_artifacts(&cfg, &report)
            .expect("special parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn array_api_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_array_api_packet(&cfg, "FSCI-P2C-007_arrayapi_core.json")
            .expect("array_api packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one array_api test case"
        );

        let artifacts = write_parity_artifacts(&cfg, &report)
            .expect("array_api parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn sparse_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_sparse_packet(&cfg, "FSCI-P2C-004_sparse_ops.json")
            .expect("sparse packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one sparse test case"
        );

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("sparse parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn fft_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_fft_packet(&cfg, "FSCI-P2C-005_fft_core.json")
            .expect("fft packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one fft test case"
        );

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("fft parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn casp_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_casp_packet(&cfg, "FSCI-P2C-008_runtime_casp.json")
            .expect("casp packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one casp test case"
        );

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("casp parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn differential_test_optimize_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-003_optimize_core.json");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle)
            .expect("differential optimize should succeed");

        assert_eq!(report.packet_id, "FSCI-P2C-003");
        assert_eq!(report.family, "optimize_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 29);
    }

    #[test]
    fn differential_optimize_quota_and_structured_logs() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-003_optimize_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read optimize fixture");
        let fixture: OptimizePacketFixture =
            serde_json::from_str(&raw).expect("parse optimize fixture");
        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("optimize differential runs");

        let mut by_case = std::collections::BTreeMap::new();
        for case in &fixture.cases {
            by_case.insert(case.case_id().to_owned(), case);
        }

        let mut differential_count = 0usize;
        let mut metamorphic_count = 0usize;
        let mut adversarial_count = 0usize;
        let mut logs = Vec::with_capacity(report.per_case_results.len());

        for case_result in &report.per_case_results {
            let case = by_case
                .get(&case_result.case_id)
                .expect("every report case should exist in fixture");
            match case.category() {
                "differential" => differential_count += 1,
                "metamorphic" => metamorphic_count += 1,
                "adversarial" => adversarial_count += 1,
                other => panic!("unexpected category: {other}"),
            }

            let log = StructuredCaseLog {
                test_id: case_result.case_id.clone(),
                category: case.category().to_owned(),
                input_summary: optimize_case_input_summary(case),
                expected: optimize_case_expected_summary(case.expected()),
                actual: case_result.message.clone(),
                diff: case_result.max_diff,
                tolerance: case_result.tolerance_used.clone(),
                pass: case_result.passed,
            };

            let payload =
                serde_json::to_string(&log).expect("structured conformance log should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&payload).expect("structured log should parse");
            assert!(parsed.get("test_id").is_some());
            assert!(parsed.get("category").is_some());
            assert!(parsed.get("input_summary").is_some());
            assert!(parsed.get("expected").is_some());
            assert!(parsed.get("actual").is_some());
            assert!(parsed.get("diff").is_some());
            assert!(parsed.get("tolerance").is_some());
            assert!(parsed.get("pass").is_some());
            logs.push(log);
        }

        assert!(
            differential_count >= 15,
            "expected >=15 differential cases, got {differential_count}"
        );
        assert!(
            metamorphic_count >= 6,
            "expected >=6 metamorphic cases, got {metamorphic_count}"
        );
        assert!(
            adversarial_count >= 8,
            "expected >=8 adversarial cases, got {adversarial_count}"
        );
        assert_eq!(report.fail_count, 0);

        let output_dir = HarnessConfig::default_paths()
            .artifact_dir_for("P2C-003")
            .join("differential");
        fs::create_dir_all(&output_dir).expect("create optimize differential artifact directory");
        let output_path = output_dir.join("structured_case_logs.json");
        let payload = serde_json::to_vec_pretty(&logs).expect("serialize optimize logs");
        fs::write(&output_path, payload).expect("write optimize structured logs");
        assert!(output_path.exists());
    }

    #[test]
    fn differential_test_special_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-006_special_core.json");
        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("differential special runs");

        assert_eq!(report.packet_id, "FSCI-P2C-006");
        assert_eq!(report.family, "special_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 29);
    }

    #[test]
    fn differential_special_quota_and_structured_logs() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-006_special_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read special fixture");
        let fixture: SpecialPacketFixture =
            serde_json::from_str(&raw).expect("parse special fixture");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("special differential");

        let mut by_case = std::collections::BTreeMap::new();
        for case in &fixture.cases {
            by_case.insert(case.case_id().to_owned(), case);
        }

        let mut differential_count = 0usize;
        let mut metamorphic_count = 0usize;
        let mut adversarial_count = 0usize;
        let mut logs = Vec::with_capacity(report.per_case_results.len());

        for case_result in &report.per_case_results {
            let case = by_case
                .get(&case_result.case_id)
                .expect("every report case should exist in fixture");
            match case.category() {
                "differential" => differential_count += 1,
                "metamorphic" => metamorphic_count += 1,
                "adversarial" => adversarial_count += 1,
                other => panic!("unexpected category: {other}"),
            }

            let log = StructuredCaseLog {
                test_id: case_result.case_id.clone(),
                category: case.category().to_owned(),
                input_summary: special_case_input_summary(case),
                expected: special_case_expected_summary(&case.expected),
                actual: case_result.message.clone(),
                diff: case_result.max_diff,
                tolerance: case_result.tolerance_used.clone(),
                pass: case_result.passed,
            };

            let payload =
                serde_json::to_string(&log).expect("structured conformance log should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&payload).expect("structured log should parse");
            assert!(parsed.get("test_id").is_some());
            assert!(parsed.get("category").is_some());
            assert!(parsed.get("input_summary").is_some());
            assert!(parsed.get("expected").is_some());
            assert!(parsed.get("actual").is_some());
            assert!(parsed.get("diff").is_some());
            assert!(parsed.get("tolerance").is_some());
            assert!(parsed.get("pass").is_some());
            logs.push(log);
        }

        assert!(
            differential_count >= 15,
            "expected >=15 differential cases, got {differential_count}"
        );
        assert!(
            metamorphic_count >= 6,
            "expected >=6 metamorphic cases, got {metamorphic_count}"
        );
        assert!(
            adversarial_count >= 8,
            "expected >=8 adversarial cases, got {adversarial_count}"
        );
        assert_eq!(report.fail_count, 0);

        let output_dir = HarnessConfig::default_paths()
            .artifact_dir_for("P2C-006")
            .join("differential");
        fs::create_dir_all(&output_dir).expect("create special differential artifact directory");
        let output_path = output_dir.join("structured_case_logs.json");
        let payload = serde_json::to_vec_pretty(&logs).expect("serialize special logs");
        fs::write(&output_path, payload).expect("write special structured logs");
        assert!(output_path.exists());
    }

    #[test]
    fn differential_test_array_api_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-007_arrayapi_core.json");
        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("differential array_api runs");

        assert_eq!(report.packet_id, "FSCI-P2C-007");
        assert_eq!(report.family, "array_api_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 29);
    }

    #[test]
    fn differential_array_api_quota_and_structured_logs() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-007_arrayapi_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read array api fixture");
        let fixture: ArrayApiPacketFixture =
            serde_json::from_str(&raw).expect("parse array api fixture");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("array api differential");

        let mut by_case = std::collections::BTreeMap::new();
        for case in &fixture.cases {
            by_case.insert(case.case_id().to_owned(), case);
        }

        let mut differential_count = 0usize;
        let mut metamorphic_count = 0usize;
        let mut adversarial_count = 0usize;
        let mut logs = Vec::with_capacity(report.per_case_results.len());

        for case_result in &report.per_case_results {
            let case = by_case
                .get(&case_result.case_id)
                .expect("every report case should exist in fixture");
            match case.category() {
                "differential" => differential_count += 1,
                "metamorphic" => metamorphic_count += 1,
                "adversarial" => adversarial_count += 1,
                other => panic!("unexpected category: {other}"),
            }

            let log = StructuredCaseLog {
                test_id: case_result.case_id.clone(),
                category: case.category().to_owned(),
                input_summary: array_api_case_input_summary(case),
                expected: array_api_case_expected_summary(case.expected()),
                actual: case_result.message.clone(),
                diff: case_result.max_diff,
                tolerance: case_result.tolerance_used.clone(),
                pass: case_result.passed,
            };

            let payload =
                serde_json::to_string(&log).expect("structured conformance log should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&payload).expect("structured log should parse");
            assert!(parsed.get("test_id").is_some());
            assert!(parsed.get("category").is_some());
            assert!(parsed.get("input_summary").is_some());
            assert!(parsed.get("expected").is_some());
            assert!(parsed.get("actual").is_some());
            assert!(parsed.get("diff").is_some());
            assert!(parsed.get("tolerance").is_some());
            assert!(parsed.get("pass").is_some());
            logs.push(log);
        }

        assert!(
            differential_count >= 15,
            "expected >=15 differential cases, got {differential_count}"
        );
        assert!(
            metamorphic_count >= 6,
            "expected >=6 metamorphic cases, got {metamorphic_count}"
        );
        assert!(
            adversarial_count >= 8,
            "expected >=8 adversarial cases, got {adversarial_count}"
        );
        assert_eq!(report.fail_count, 0);

        let output_dir = HarnessConfig::default_paths()
            .artifact_dir_for("P2C-007")
            .join("differential");
        fs::create_dir_all(&output_dir).expect("create array_api differential artifact directory");
        let output_path = output_dir.join("structured_case_logs.json");
        let payload = serde_json::to_vec_pretty(&logs).expect("serialize array api logs");
        fs::write(&output_path, payload).expect("write array_api structured logs");
        assert!(output_path.exists());
    }

    #[test]
    fn differential_test_graceful_oracle_missing() {
        let oracle = DifferentialOracleConfig {
            python_path: PathBuf::from("/nonexistent/python"),
            script_path: PathBuf::from("/nonexistent/script.py"),
            timeout_secs: 5,
            required: false,
        };
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-001_validate_tol.json");

        let report = run_differential_test(&fixture_path, &oracle)
            .expect("should succeed even with missing oracle");

        // Report should still pass against embedded expected values
        assert_eq!(report.fail_count, 0);
        assert!(matches!(report.oracle_status, OracleStatus::Missing { .. }));
    }

    #[test]
    fn differential_test_unknown_family_errors() {
        let unique = format!("fsci-diff-unknown-{}", super::now_unix_ms());
        let dir = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&dir).expect("create temp dir");
        let fixture_path = dir.join("bad_fixture.json");
        fs::write(
            &fixture_path,
            r#"{"packet_id":"FSCI-P2C-999","family":"unknown_type","cases":[]}"#,
        )
        .expect("write fixture");

        let oracle = default_test_oracle();
        let result = run_differential_test(&fixture_path, &oracle);
        assert!(result.is_err(), "unknown family should produce an error");
    }

    #[test]
    fn conformance_report_serializes_to_json() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-002_linalg_core.json");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("test should succeed");

        let json = serde_json::to_string_pretty(&report).expect("report must serialize");
        let parsed: ConformanceReport =
            serde_json::from_str(&json).expect("report must round-trip");
        assert_eq!(parsed.packet_id, report.packet_id);
        assert_eq!(parsed.pass_count, report.pass_count);
        assert_eq!(parsed.per_case_results.len(), report.per_case_results.len());
    }

    // ═══════════════════════════════════════════════════════════════
    // Packet runner registry tests (§bd-3jh.19.10)
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn packet_family_all_has_8_entries() {
        assert_eq!(PacketFamily::ALL.len(), 8);
    }

    #[test]
    fn packet_family_ids_are_unique() {
        let mut ids: Vec<&str> = PacketFamily::ALL.iter().map(|f| f.packet_id()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), PacketFamily::ALL.len());
    }

    #[test]
    fn packet_family_names_are_unique() {
        let mut names: Vec<&str> = PacketFamily::ALL.iter().map(|f| f.family_name()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), PacketFamily::ALL.len());
    }

    #[test]
    fn packet_family_from_family_str_roundtrip() {
        for family in PacketFamily::ALL {
            let detected = PacketFamily::from_family_str(family.family_name());
            assert_eq!(
                detected,
                Some(family),
                "from_family_str should roundtrip for {}",
                family.family_name()
            );
        }
    }

    #[test]
    fn packet_family_from_family_str_unknown() {
        assert_eq!(PacketFamily::from_family_str("completely_unknown"), None);
    }

    #[test]
    fn discover_fixtures_finds_existing() {
        let config = HarnessConfig::default_paths();
        let fixtures = discover_fixtures(&config).expect("discover fixtures");
        assert!(
            fixtures.len() >= 3,
            "should find at least P2C-001, P2C-002, and P2C-003 fixtures"
        );
        let families: Vec<PacketFamily> = fixtures.iter().map(|(f, _)| *f).collect();
        assert!(families.contains(&PacketFamily::ValidateTol));
        assert!(families.contains(&PacketFamily::LinalgCore));
        assert!(families.contains(&PacketFamily::Optimize));
    }

    #[test]
    fn ensure_artifact_layout_creates_dirs() {
        let unique = format!("fsci-layout-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(root.join("fixtures")).expect("create root");
        let config = HarnessConfig {
            oracle_root: PathBuf::new(),
            fixture_root: root.join("fixtures"),
            strict_mode: true,
        };

        let base = ensure_artifact_layout(&config, "P2C-008").expect("create layout");
        assert!(base.join("anchor").exists());
        assert!(base.join("contracts").exists());
        assert!(base.join("threats").exists());

        // Idempotent: calling again should succeed
        ensure_artifact_layout(&config, "P2C-008").expect("idempotent layout");
    }

    #[test]
    fn aggregate_report_sums_correctly() {
        let make_cases = |n: usize, pass: bool| -> Vec<super::CaseResult> {
            (0..n)
                .map(|i| super::CaseResult {
                    case_id: format!("case_{i}"),
                    passed: pass,
                    message: String::new(),
                })
                .collect()
        };

        let mut cases_1 = make_cases(10, true);
        cases_1.extend(make_cases(2, false));
        let mut cases_2 = make_cases(20, true);
        cases_2.extend(make_cases(1, false));

        let reports = vec![
            super::PacketReport {
                packet_id: "FSCI-P2C-001".to_owned(),
                family: "validate_tol".to_owned(),
                case_results: cases_1,
                passed_cases: 10,
                failed_cases: 2,
                generated_unix_ms: 0,
            },
            super::PacketReport {
                packet_id: "FSCI-P2C-002".to_owned(),
                family: "linalg_core".to_owned(),
                case_results: cases_2,
                passed_cases: 20,
                failed_cases: 1,
                generated_unix_ms: 0,
            },
        ];

        let agg = aggregate_packet_reports(&reports);
        assert_eq!(agg.total_cases, 33);
        assert_eq!(agg.total_passed, 30);
        assert_eq!(agg.total_failed, 3);
        assert_eq!(agg.packets.len(), 2);
    }

    #[test]
    fn aggregate_report_serializes() {
        let agg = AggregateParityReport {
            generated_unix_ms: 1234567890,
            packets: vec![],
            total_cases: 0,
            total_passed: 0,
            total_failed: 0,
        };
        let json = serde_json::to_string(&agg).expect("serialize");
        let parsed: AggregateParityReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.generated_unix_ms, 1234567890);
    }

    #[test]
    fn packet_family_fixture_filename() {
        assert_eq!(
            PacketFamily::ValidateTol.fixture_filename(),
            "FSCI-P2C-001_validate_tol.json"
        );
        assert_eq!(
            PacketFamily::RuntimeCasp.fixture_filename(),
            "FSCI-P2C-008_runtime_casp.json"
        );
    }

    #[test]
    fn packet_family_has_runner() {
        assert!(PacketFamily::ValidateTol.has_runner());
        assert!(PacketFamily::LinalgCore.has_runner());
        assert!(PacketFamily::Optimize.has_runner());
        assert!(PacketFamily::SparseOps.has_runner());
        assert!(PacketFamily::Fft.has_runner());
        assert!(PacketFamily::Special.has_runner());
        assert!(PacketFamily::ArrayApi.has_runner());
        assert!(PacketFamily::RuntimeCasp.has_runner());
    }
}
