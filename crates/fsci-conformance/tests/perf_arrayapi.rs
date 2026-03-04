//! P2C-007-H: Performance profiling for Array API broadcast hot path.
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
//! "before" uses the legacy per-element unravel/ravel algorithm.
//! "after" uses the current optimized `CoreArrayBackend::broadcast_to`.

use fsci_arrayapi::{
    ArrayApiArray, ArrayApiBackend, CoreArray, CoreArrayBackend, CreationRequest, DType,
    ExecutionMode, MemoryOrder, ScalarValue, Shape, from_slice,
};
use serde::Serialize;
use std::time::Instant;

const SIZES: &[usize] = &[10, 100, 1000, 10_000];
const DTYPES: &[DType] = &[DType::Float32, DType::Float64, DType::Complex128];
const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 30;
const HOTSPOT: &str = "CoreArrayBackend::broadcast_to";

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

#[derive(Serialize)]
struct IsomorphismCheck {
    all_cases_pass: bool,
    details: Vec<IsomorphismDetail>,
}

#[derive(Serialize)]
struct IsomorphismDetail {
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

#[test]
fn perf_p2c007_arrayapi_broadcast_profile() {
    let backend = strict_backend();
    let mut rows = Vec::new();
    let mut iso_details = Vec::new();

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
                array_size: size,
                dtype: format!("{dtype:?}"),
                passes: isomorphic,
                note,
            });

            let before_p95_i128 =
                i128::try_from(before_stats.p95_ns).expect("before p95 should fit i128");
            let after_p95_i128 =
                i128::try_from(after_stats.p95_ns).expect("after p95 should fit i128");

            rows.push(BenchmarkRow {
                hotspot_function: HOTSPOT.to_string(),
                array_size: size,
                output_elements,
                dtype: format!("{dtype:?}"),
                before_p95_ns: before_stats.p95_ns,
                after_p95_ns: after_stats.p95_ns,
                before_median_ns: before_stats.median_ns,
                after_median_ns: after_stats.median_ns,
                p95_improvement_ns: before_p95_i128 - after_p95_i128,
                alloc_count_delta: estimated_alloc_count_delta(output_elements),
            });
        }
    }

    let all_pass = iso_details.iter().all(|entry| entry.passes);
    assert!(all_pass, "broadcast optimization changed observable values");

    let report = PerfReport {
        generated_at: chrono_lite_now(),
        optimization_name: "incremental row-major coordinate advancement".to_string(),
        hotspot_function: HOTSPOT.to_string(),
        benchmark_rows: rows,
        isomorphism_check: IsomorphismCheck {
            all_cases_pass: all_pass,
            details: iso_details,
        },
        methodology: vec![
            format!("warmup_iters={WARMUP_ITERS}"),
            format!("bench_iters={BENCH_ITERS}"),
            "before=legacy per-element unravel/ravel + per-iteration coordinate Vec allocations"
                .to_string(),
            "after=CoreArrayBackend::broadcast_to with precomputed strides and in-place coordinate advancement"
                .to_string(),
            "alloc_count_delta is an algorithmic estimate (after - before), not allocator-instrumented"
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
