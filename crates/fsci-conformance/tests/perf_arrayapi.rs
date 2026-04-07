//! P2C-007-H: Performance profiling for Array API broadcast and creation hot paths.
//!
//! Produces structured JSON artifact at:
//!   fixtures/artifacts/P2C-007/perf/perf_profile_report.json
//!
//! The report includes the required log fields:
//! - hotspot_function
//! - array_size
//! - dtype
//! - before_p95_ns
//! - after_p95_ns
//! - alloc_count_delta
//!
//! Broadcast "before" uses the legacy per-element unravel/ravel algorithm.
//! Broadcast "after" uses the current optimized `CoreArrayBackend::broadcast_to`.
//! Creation "before" uses the current `CoreArrayBackend` path with hot-path
//! dtype dispatch logging still enabled.
//! Creation "after" uses a local log-free profiling backend that matches the
//! observable array contract for the scoped creation cases.

use fsci_arrayapi::{
    ArrayApiArray, ArrayApiBackend, ArrayApiError, ArrayApiErrorKind, ArrayApiResult, CoreArray,
    CoreArrayBackend, CreationRequest, DType, ExecutionMode, FullRequest, MemoryOrder, ScalarValue,
    Shape, from_slice, full, zeros,
};
use serde::Serialize;
use std::time::Instant;

const SIZES: &[usize] = &[10, 100, 1000, 10_000];
const DTYPES: &[DType] = &[DType::Float32, DType::Float64, DType::Complex128];
const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 30;
const HOTSPOT_ARRAYAPI: &str = "fsci-arrayapi hotpath portfolio";
const HOTSPOT_BROADCAST: &str = "CoreArrayBackend::broadcast_to";
const HOTSPOT_ZEROS: &str = "CoreArrayBackend::zeros";
const HOTSPOT_FULL: &str = "CoreArrayBackend::full";

#[derive(Serialize)]
struct PerfReport {
    generated_at: String,
    optimization_name: String,
    hotspot_function: String,
    benchmark_rows: Vec<BenchmarkRow>,
    isomorphism_check: IsomorphismCheck,
    methodology: Vec<String>,
}

#[derive(Serialize)]
struct BenchmarkRow {
    hotspot_function: String,
    array_size: usize,
    output_elements: usize,
    dtype: String,
    before_p95_ns: u128,
    after_p95_ns: u128,
    before_median_ns: u128,
    after_median_ns: u128,
    p95_improvement_ns: i128,
    alloc_count_delta: i64,
}

impl BenchmarkRow {
    fn new(
        hotspot_function: &str,
        array_size: usize,
        output_elements: usize,
        dtype: DType,
        before_stats: BenchStats,
        after_stats: BenchStats,
        alloc_count_delta: i64,
    ) -> Self {
        let before_p95_i128 =
            i128::try_from(before_stats.p95_ns).expect("before p95 should fit i128");
        let after_p95_i128 = i128::try_from(after_stats.p95_ns).expect("after p95 should fit i128");

        Self {
            hotspot_function: hotspot_function.to_string(),
            array_size,
            output_elements,
            dtype: format!("{dtype:?}"),
            before_p95_ns: before_stats.p95_ns,
            after_p95_ns: after_stats.p95_ns,
            before_median_ns: before_stats.median_ns,
            after_median_ns: after_stats.median_ns,
            p95_improvement_ns: before_p95_i128 - after_p95_i128,
            alloc_count_delta,
        }
    }
}

#[derive(Serialize)]
struct IsomorphismCheck {
    all_cases_pass: bool,
    details: Vec<IsomorphismDetail>,
}

#[derive(Serialize)]
struct IsomorphismDetail {
    hotspot_function: String,
    array_size: usize,
    dtype: String,
    passes: bool,
    note: String,
}

#[derive(Clone, Copy)]
struct BenchStats {
    median_ns: u128,
    p95_ns: u128,
}

fn strict_backend() -> CoreArrayBackend {
    CoreArrayBackend::new(ExecutionMode::Strict)
}

#[derive(Debug, Clone)]
struct ProfileArray {
    shape: Shape,
    dtype: DType,
    values: Vec<ScalarValue>,
}

impl ProfileArray {
    fn values(&self) -> &[ScalarValue] {
        &self.values
    }
}

impl ArrayApiArray for ProfileArray {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

#[derive(Debug, Default)]
struct ProfileArrayBackend;

impl ArrayApiBackend for ProfileArrayBackend {
    type Array = ProfileArray;

    fn namespace_name(&self) -> &'static str {
        "profile_array_api"
    }

    fn shape_of(&self, array: &Self::Array) -> Shape {
        array.shape.clone()
    }

    fn dtype_of(&self, array: &Self::Array) -> DType {
        array.dtype
    }

    fn asarray(
        &self,
        value: ScalarValue,
        dtype: Option<DType>,
        _copy: Option<bool>,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = profile_resolve_dtype(dtype)?;
        Ok(ProfileArray {
            shape: Shape::scalar(),
            dtype: resolved_dtype,
            values: vec![profile_cast_scalar(value, resolved_dtype)?],
        })
    }

    fn zeros(
        &self,
        shape: &Shape,
        dtype: DType,
        _order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        profile_filled_array(shape, ScalarValue::F64(0.0), dtype)
    }

    fn ones(
        &self,
        shape: &Shape,
        dtype: DType,
        _order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        profile_filled_array(shape, ScalarValue::F64(1.0), dtype)
    }

    fn empty(
        &self,
        shape: &Shape,
        dtype: DType,
        _order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        profile_filled_array(shape, ScalarValue::F64(0.0), dtype)
    }

    fn full(
        &self,
        shape: &Shape,
        fill_value: ScalarValue,
        dtype: DType,
        _order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        profile_filled_array(shape, fill_value, dtype)
    }

    fn arange(
        &self,
        _start: ScalarValue,
        _stop: ScalarValue,
        _step: ScalarValue,
        _dtype: Option<DType>,
    ) -> ArrayApiResult<Self::Array> {
        profile_unimplemented("arange")
    }

    fn linspace(
        &self,
        _start: ScalarValue,
        _stop: ScalarValue,
        _num: usize,
        _endpoint: bool,
        _dtype: Option<DType>,
    ) -> ArrayApiResult<Self::Array> {
        profile_unimplemented("linspace")
    }

    fn getitem(
        &self,
        _array: &Self::Array,
        _index: &fsci_arrayapi::IndexExpr,
    ) -> ArrayApiResult<Self::Array> {
        profile_unimplemented("getitem")
    }

    fn broadcast_to(&self, _array: &Self::Array, _shape: &Shape) -> ArrayApiResult<Self::Array> {
        profile_unimplemented("broadcast_to")
    }

    fn astype(&self, _array: &Self::Array, _dtype: DType) -> ArrayApiResult<Self::Array> {
        profile_unimplemented("astype")
    }

    fn result_type(&self, _dtypes: &[DType], _force_floating: bool) -> ArrayApiResult<DType> {
        profile_unimplemented("result_type")
    }
}

fn make_sequence_values(len: usize) -> Vec<ScalarValue> {
    (0..len)
        .map(|idx| ScalarValue::F64((idx as f64) * 0.25 + 1.0))
        .collect()
}

fn make_array(backend: &CoreArrayBackend, shape: Shape, dtype: DType) -> CoreArray {
    let values = make_sequence_values(shape.element_count().expect("shape must not overflow"));
    let request = CreationRequest {
        shape,
        dtype,
        order: MemoryOrder::C,
    };
    from_slice(backend, &values, &request).expect("array construction should succeed")
}

fn profile_resolve_dtype(dtype: Option<DType>) -> ArrayApiResult<DType> {
    let resolved_dtype = dtype.unwrap_or(DType::Float64);
    match resolved_dtype {
        DType::Float32 | DType::Float64 | DType::Complex128 => Ok(resolved_dtype),
        _ => Err(ArrayApiError::new(
            ArrayApiErrorKind::UnsupportedDtype,
            "profile backend only covers Float32/Float64/Complex128",
        )),
    }
}

fn profile_scalar_to_f64(value: ScalarValue) -> ArrayApiResult<f64> {
    match value {
        ScalarValue::Bool(v) => Ok(if v { 1.0 } else { 0.0 }),
        ScalarValue::I64(v) => Ok(v as f64),
        ScalarValue::U64(v) => Ok(v as f64),
        ScalarValue::F64(v) => Ok(v),
        ScalarValue::ComplexF64 { re, im } => {
            if im == 0.0 {
                Ok(re)
            } else {
                Err(ArrayApiError::new(
                    ArrayApiErrorKind::UnsupportedDtype,
                    "profile backend does not coerce complex values with nonzero imaginary part",
                ))
            }
        }
    }
}

fn profile_cast_scalar(value: ScalarValue, dtype: DType) -> ArrayApiResult<ScalarValue> {
    match dtype {
        DType::Float32 => Ok(ScalarValue::F64(
            (profile_scalar_to_f64(value)? as f32) as f64,
        )),
        DType::Float64 => Ok(ScalarValue::F64(profile_scalar_to_f64(value)?)),
        DType::Complex128 => Ok(ScalarValue::ComplexF64 {
            re: profile_scalar_to_f64(value)?,
            im: 0.0,
        }),
        _ => Err(ArrayApiError::new(
            ArrayApiErrorKind::UnsupportedDtype,
            "profile backend only covers Float32/Float64/Complex128",
        )),
    }
}

fn profile_filled_array(
    shape: &Shape,
    fill_value: ScalarValue,
    dtype: DType,
) -> ArrayApiResult<ProfileArray> {
    let resolved_dtype = profile_resolve_dtype(Some(dtype))?;
    let size = shape.element_count().ok_or_else(|| {
        ArrayApiError::new(ArrayApiErrorKind::Overflow, "shape element count overflow")
    })?;
    let fill = profile_cast_scalar(fill_value, resolved_dtype)?;
    Ok(ProfileArray {
        shape: shape.clone(),
        dtype: resolved_dtype,
        values: vec![fill; size],
    })
}

fn profile_unimplemented<T>(operation: &'static str) -> ArrayApiResult<T> {
    Err(ArrayApiError::new(
        ArrayApiErrorKind::NotYetImplemented,
        operation,
    ))
}

fn legacy_unravel_index(mut index: usize, dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }
    let mut coords = vec![0usize; dims.len()];
    for pos in (0..dims.len()).rev() {
        let dim = dims[pos];
        if dim == 0 {
            coords[pos] = 0;
        } else {
            coords[pos] = index % dim;
            index /= dim;
        }
    }
    coords
}

fn legacy_ravel_index(coords: &[usize], dims: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut index = 0usize;
    for pos in (0..dims.len()).rev() {
        index += coords[pos] * stride;
        stride = stride.saturating_mul(dims[pos].max(1));
    }
    index
}

fn legacy_broadcast_values(array: &CoreArray, out_shape: &Shape) -> Vec<ScalarValue> {
    let output_size = out_shape
        .element_count()
        .expect("broadcast output shape must not overflow");
    let out_rank = out_shape.rank();
    let in_shape = array.shape();
    let in_rank = in_shape.rank();
    let mut values = Vec::with_capacity(output_size);

    for linear in 0..output_size {
        let out_coords = legacy_unravel_index(linear, &out_shape.dims);
        let mut in_coords = vec![0usize; in_rank];
        for (in_dim_idx, in_dim) in in_shape.dims.iter().enumerate() {
            let out_dim_idx = out_rank - in_rank + in_dim_idx;
            in_coords[in_dim_idx] = if *in_dim == 1 {
                0
            } else {
                out_coords[out_dim_idx]
            };
        }
        let in_linear = legacy_ravel_index(&in_coords, &in_shape.dims);
        values.push(array.values()[in_linear]);
    }

    values
}

fn time_operation<F: FnMut()>(mut f: F) -> BenchStats {
    for _ in 0..WARMUP_ITERS {
        f();
    }

    let mut timings = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        f();
        timings.push(start.elapsed().as_nanos());
    }
    timings.sort_unstable();

    let median_idx = timings.len() / 2;
    let p95_idx = (timings.len() * 95).saturating_sub(1) / 100;

    BenchStats {
        median_ns: timings[median_idx],
        p95_ns: timings[p95_idx],
    }
}

fn estimated_alloc_count_delta(output_elements: usize) -> i64 {
    // Legacy loop: 1 values Vec + 2 temporary Vec allocations per output element.
    // Optimized loop: values Vec + in_strides Vec + out_coords Vec.
    // Delta is (after - before) in allocation events.
    let before = 1usize.saturating_add(output_elements.saturating_mul(2));
    let after = 3usize;
    let before_i64 = i64::try_from(before).expect("before allocation estimate fits i64");
    let after_i64 = i64::try_from(after).expect("after allocation estimate fits i64");
    after_i64 - before_i64
}

fn bench_broadcast_profile(rows: &mut Vec<BenchmarkRow>, iso_details: &mut Vec<IsomorphismDetail>) {
    let backend = strict_backend();

    for &dtype in DTYPES {
        for &size in SIZES {
            let input = make_array(&backend, Shape::new(vec![size, 1]), dtype);
            let out_shape = Shape::new(vec![size, 2]);
            let output_elements = out_shape.element_count().expect("shape must not overflow");

            let before_stats = time_operation(|| {
                let _ = legacy_broadcast_values(&input, &out_shape);
            });
            let after_stats = time_operation(|| {
                let _ = backend
                    .broadcast_to(&input, &out_shape)
                    .expect("optimized broadcast should succeed");
            });

            let legacy_values = legacy_broadcast_values(&input, &out_shape);
            let optimized = backend
                .broadcast_to(&input, &out_shape)
                .expect("optimized broadcast should succeed");
            let isomorphic = legacy_values == optimized.values();
            let note = if isomorphic {
                "legacy and optimized outputs are byte-identical".to_string()
            } else {
                format!(
                    "value mismatch: legacy_len={}, optimized_len={}",
                    legacy_values.len(),
                    optimized.values().len()
                )
            };
            iso_details.push(IsomorphismDetail {
                hotspot_function: HOTSPOT_BROADCAST.to_string(),
                array_size: size,
                dtype: format!("{dtype:?}"),
                passes: isomorphic,
                note,
            });

            rows.push(BenchmarkRow::new(
                HOTSPOT_BROADCAST,
                size,
                output_elements,
                dtype,
                before_stats,
                after_stats,
                estimated_alloc_count_delta(output_elements),
            ));
        }
    }
}

fn bench_creation_profile(rows: &mut Vec<BenchmarkRow>, iso_details: &mut Vec<IsomorphismDetail>) {
    let backend = strict_backend();
    let profile_backend = ProfileArrayBackend;

    for &dtype in DTYPES {
        for &size in SIZES {
            let request = CreationRequest {
                shape: Shape::new(vec![size]),
                dtype,
                order: MemoryOrder::C,
            };
            let full_request = FullRequest {
                fill_value: ScalarValue::F64(3.25),
                dtype,
                order: MemoryOrder::C,
            };

            let before_zero = time_operation(|| {
                let _ = zeros(&backend, &request).expect("zeros should succeed");
            });
            let after_zero = time_operation(|| {
                let _ = zeros(&profile_backend, &request).expect("profile zeros should succeed");
            });

            let current_zero = zeros(&backend, &request).expect("zeros should succeed");
            let profile_zero =
                zeros(&profile_backend, &request).expect("profile zeros should succeed");
            let zero_isomorphic = current_zero.shape() == profile_zero.shape()
                && current_zero.dtype() == profile_zero.dtype()
                && current_zero.values() == profile_zero.values();
            let zero_note = if zero_isomorphic {
                "current and log-free zeros paths match exactly".to_string()
            } else {
                format!(
                    "zeros mismatch: current_len={}, profile_len={}",
                    current_zero.values().len(),
                    profile_zero.values().len()
                )
            };
            iso_details.push(IsomorphismDetail {
                hotspot_function: HOTSPOT_ZEROS.to_string(),
                array_size: size,
                dtype: format!("{dtype:?}"),
                passes: zero_isomorphic,
                note: zero_note,
            });
            rows.push(BenchmarkRow::new(
                HOTSPOT_ZEROS,
                size,
                size,
                dtype,
                before_zero,
                after_zero,
                0,
            ));

            let before_full = time_operation(|| {
                let _ = full(&backend, &request.shape, &full_request).expect("full should succeed");
            });
            let after_full = time_operation(|| {
                let _ = full(&profile_backend, &request.shape, &full_request)
                    .expect("profile full should succeed");
            });

            let current_full =
                full(&backend, &request.shape, &full_request).expect("full should succeed");
            let profile_full = full(&profile_backend, &request.shape, &full_request)
                .expect("profile full should succeed");
            let full_isomorphic = current_full.shape() == profile_full.shape()
                && current_full.dtype() == profile_full.dtype()
                && current_full.values() == profile_full.values();
            let full_note = if full_isomorphic {
                "current and log-free full paths match exactly".to_string()
            } else {
                format!(
                    "full mismatch: current_len={}, profile_len={}",
                    current_full.values().len(),
                    profile_full.values().len()
                )
            };
            iso_details.push(IsomorphismDetail {
                hotspot_function: HOTSPOT_FULL.to_string(),
                array_size: size,
                dtype: format!("{dtype:?}"),
                passes: full_isomorphic,
                note: full_note,
            });
            rows.push(BenchmarkRow::new(
                HOTSPOT_FULL,
                size,
                size,
                dtype,
                before_full,
                after_full,
                0,
            ));
        }
    }
}

#[test]
fn perf_p2c007_arrayapi_hotpath_profile() {
    let mut rows = Vec::new();
    let mut iso_details = Vec::new();

    bench_broadcast_profile(&mut rows, &mut iso_details);
    bench_creation_profile(&mut rows, &mut iso_details);

    let all_pass = iso_details.iter().all(|entry| entry.passes);
    assert!(
        all_pass,
        "arrayapi perf characterization changed observable values"
    );

    let report = PerfReport {
        generated_at: chrono_lite_now(),
        optimization_name: "broadcast optimization plus creation-path characterization"
            .to_string(),
        hotspot_function: HOTSPOT_ARRAYAPI.to_string(),
        benchmark_rows: rows,
        isomorphism_check: IsomorphismCheck {
            all_cases_pass: all_pass,
            details: iso_details,
        },
        methodology: vec![
            format!("warmup_iters={WARMUP_ITERS}"),
            format!("bench_iters={BENCH_ITERS}"),
            "broadcast before=legacy per-element unravel/ravel + per-iteration coordinate Vec allocations"
                .to_string(),
            "broadcast after=CoreArrayBackend::broadcast_to with precomputed strides and in-place coordinate advancement"
                .to_string(),
            "creation before=current CoreArrayBackend creation path with dtype dispatch logging enabled"
                .to_string(),
            "creation after=local log-free profiling backend matching scoped observable creation semantics"
                .to_string(),
            "broadcast alloc_count_delta is an algorithmic estimate (after - before), not allocator-instrumented"
                .to_string(),
            "creation alloc_count_delta is set to 0 because the characterization isolates hot-path logging overhead rather than allocator-instrumented buffer counts"
                .to_string(),
        ],
    };

    let artifact_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/P2C-007/perf");
    std::fs::create_dir_all(&artifact_dir).expect("perf artifact directory should be creatable");

    let pretty_json = serde_json::to_string_pretty(&report).expect("perf report should serialize");
    std::fs::write(artifact_dir.join("perf_profile_report.json"), pretty_json)
        .expect("perf profile report should be written");

    // Emit a machine-readable line so remote test runs can reconstruct the report locally.
    let compact_json =
        serde_json::to_string(&report).expect("compact perf report should serialize");
    println!("P2C007_PERF_REPORT_JSON={compact_json}");
}

/// Minimal timestamp without pulling in chrono.
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_secs();
    format!("unix:{secs}")
}
