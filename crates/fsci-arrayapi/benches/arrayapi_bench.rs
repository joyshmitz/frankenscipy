use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_arrayapi::{
    ArrayApiArray, ArrayApiBackend, CoreArray, CoreArrayBackend, CreationRequest, DType,
    ExecutionMode, FullRequest, IndexExpr, IndexRequest, IndexingMode, MemoryOrder, ScalarValue,
    Shape, SliceSpec, from_slice, full, getitem, promote_and_broadcast, zeros,
};
use nalgebra::DMatrix;
use std::hint::black_box;

const SIZES: &[usize] = &[10, 100, 1000, 10_000];
const DTYPES: &[DType] = &[DType::Float32, DType::Float64, DType::Complex128];

fn strict_backend() -> CoreArrayBackend {
    CoreArrayBackend::new(ExecutionMode::Strict)
}

fn make_sequence_values(len: usize) -> Vec<ScalarValue> {
    (0..len)
        .map(|idx| ScalarValue::F64((idx as f64) * 0.5 + 1.0))
        .collect()
}

fn make_array(
    backend: &CoreArrayBackend,
    shape: Shape,
    dtype: DType,
    order: MemoryOrder,
) -> fsci_arrayapi::CoreArray {
    let values = make_sequence_values(shape.element_count().expect("shape should not overflow"));
    let request = CreationRequest {
        shape,
        dtype,
        order,
    };
    from_slice(backend, &values, &request).expect("array construction should succeed")
}

fn to_dmatrix_original_float64(array: &CoreArray) -> DMatrix<f64> {
    let rows = array.shape().dims[0];
    let cols = array.shape().dims[1];
    let mut data = Vec::with_capacity(rows.saturating_mul(cols));
    for value in array.values() {
        let ScalarValue::F64(value) = *value else {
            panic!("benchmark input must contain float scalars"); // ubs:ignore — setup constructs a Float64 CoreArray
        };
        data.push(value);
    }
    match array.order() {
        MemoryOrder::F => DMatrix::from_column_slice(rows, cols, &data),
        _ => DMatrix::from_row_slice(rows, cols, &data),
    }
}

fn bench_to_dmatrix_ab(c: &mut Criterion) {
    let backend = strict_backend();
    let array = make_array(
        &backend,
        Shape::new(vec![512, 512]),
        DType::Float64,
        MemoryOrder::F,
    );
    let current = array.to_dmatrix().expect("direct matrix conversion");
    let original = to_dmatrix_original_float64(&array);
    assert!(
        current
            .as_slice()
            .iter()
            .zip(original.as_slice())
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
    );

    let mut group = c.benchmark_group("arrayapi_to_dmatrix_ab");
    group.bench_function("current_direct/512x512_f", |b| {
        b.iter(|| black_box(array.to_dmatrix().expect("direct matrix conversion")))
    });
    group.bench_function("original_staged/512x512_f", |b| {
        b.iter(|| black_box(to_dmatrix_original_float64(black_box(&array))))
    });
    group.finish();
}

fn bench_creation(c: &mut Criterion) {
    let backend = strict_backend();
    let mut group = c.benchmark_group("arrayapi_creation");

    for &dtype in DTYPES {
        for &size in SIZES {
            group.bench_with_input(
                BenchmarkId::new(format!("zeros_{dtype:?}"), size),
                &size,
                |b, &len| {
                    let request = CreationRequest {
                        shape: Shape::new(vec![len]),
                        dtype,
                        order: MemoryOrder::C,
                    };
                    b.iter(|| {
                        let out = zeros(&backend, black_box(&request)).expect("zeros must succeed");
                        black_box(out.size());
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("full_{dtype:?}"), size),
                &size,
                |b, &len| {
                    let shape = Shape::new(vec![len]);
                    let request = FullRequest {
                        fill_value: ScalarValue::F64(3.25),
                        dtype,
                        order: MemoryOrder::C,
                    };
                    b.iter(|| {
                        let out = full(&backend, black_box(&shape), black_box(&request))
                            .expect("full must succeed");
                        black_box(out.size());
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_broadcast(c: &mut Criterion) {
    let backend = strict_backend();
    let mut group = c.benchmark_group("arrayapi_broadcast");

    for &size in SIZES {
        let left = make_array(
            &backend,
            Shape::new(vec![size, 1]),
            DType::Float32,
            MemoryOrder::C,
        );
        let right = make_array(
            &backend,
            Shape::new(vec![1, 2]),
            DType::Complex128,
            MemoryOrder::C,
        );

        group.bench_with_input(
            BenchmarkId::new("promote_and_broadcast", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let out = promote_and_broadcast(
                        &backend,
                        &[black_box(&left), black_box(&right)],
                        false,
                    )
                    .expect("promote_and_broadcast should succeed");
                    black_box(out[0].size() + out[1].size());
                });
            },
        );
    }

    group.finish();
}

fn bench_indexing(c: &mut Criterion) {
    let backend = strict_backend();
    let mut group = c.benchmark_group("arrayapi_indexing");

    for &size in SIZES {
        let rank1_array = make_array(
            &backend,
            Shape::new(vec![size]),
            DType::Float64,
            MemoryOrder::C,
        );
        let rank1_request = IndexRequest {
            mode: IndexingMode::Basic,
            index: IndexExpr::Basic {
                slices: vec![SliceSpec {
                    start: Some(1),
                    stop: Some(size as isize - 1),
                    step: 1,
                }],
            },
        };

        group.bench_with_input(
            BenchmarkId::new("getitem_basic_rank1", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let out = getitem(&backend, black_box(&rank1_array), black_box(&rank1_request))
                        .expect("rank-1 getitem should succeed");
                    black_box(out.size());
                });
            },
        );

        let array = make_array(
            &backend,
            Shape::new(vec![size, 4]),
            DType::Float64,
            MemoryOrder::C,
        );
        let request = IndexRequest {
            mode: IndexingMode::Basic,
            index: IndexExpr::Basic {
                slices: vec![
                    SliceSpec {
                        start: Some(0),
                        stop: Some(size as isize),
                        step: 1,
                    },
                    SliceSpec {
                        start: Some(1),
                        stop: Some(3),
                        step: 1,
                    },
                ],
            },
        };

        group.bench_with_input(BenchmarkId::new("getitem_basic", size), &size, |b, _| {
            b.iter(|| {
                let out = getitem(&backend, black_box(&array), black_box(&request))
                    .expect("getitem should succeed");
                black_box(out.size());
            });
        });
    }

    let strided_size = 10_000;
    let strided_array = make_array(
        &backend,
        Shape::new(vec![strided_size, 64]),
        DType::Float64,
        MemoryOrder::C,
    );
    let strided_request = IndexRequest {
        mode: IndexingMode::Basic,
        index: IndexExpr::Basic {
            slices: vec![
                SliceSpec {
                    start: Some(0),
                    stop: Some(strided_size as isize),
                    step: 2,
                },
                SliceSpec {
                    start: Some(1),
                    stop: Some(63),
                    step: 1,
                },
            ],
        },
    };
    group.bench_function("getitem_basic_strided_rows", |b| {
        b.iter(|| {
            let out = getitem(
                &backend,
                black_box(&strided_array),
                black_box(&strided_request),
            )
            .expect("strided-row getitem should succeed");
            black_box(out.size());
        });
    });

    let strided_auto_array = make_array(
        &backend,
        Shape::new(vec![strided_size, 64]),
        DType::Float64,
        MemoryOrder::A,
    );
    assert_eq!(strided_auto_array.order(), MemoryOrder::A);
    group.bench_function("getitem_basic_strided_rows_auto", |b| {
        b.iter(|| {
            let out = getitem(
                &backend,
                black_box(&strided_auto_array),
                black_box(&strided_request),
            )
            .expect("auto-order strided-row getitem should succeed");
            black_box(out.size());
        });
    });

    let fortran_single_column_array = make_array(
        &backend,
        Shape::new(vec![strided_size, 64]),
        DType::Float64,
        MemoryOrder::F,
    );
    let fortran_single_column_request = IndexRequest {
        mode: IndexingMode::Basic,
        index: IndexExpr::Basic {
            slices: vec![
                SliceSpec {
                    start: Some(1),
                    stop: Some(strided_size as isize - 1),
                    step: 1,
                },
                SliceSpec {
                    start: Some(31),
                    stop: Some(32),
                    step: 1,
                },
            ],
        },
    };
    group.bench_function("getitem_basic_fortran_single_column", |b| {
        b.iter(|| {
            let out = getitem(
                &backend,
                black_box(&fortran_single_column_array),
                black_box(&fortran_single_column_request),
            )
            .expect("F-order single-column getitem should succeed");
            black_box(out.size());
        });
    });

    group.finish();
}

fn bench_dtype_cast(c: &mut Criterion) {
    let backend = strict_backend();
    let mut group = c.benchmark_group("arrayapi_dtype_cast");

    for &size in SIZES {
        let source = make_array(
            &backend,
            Shape::new(vec![size]),
            DType::Float32,
            MemoryOrder::C,
        );

        group.bench_with_input(
            BenchmarkId::new("astype_float32_to_complex128", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let out = backend
                        .astype(black_box(&source), DType::Complex128)
                        .expect("astype should succeed");
                    black_box(out.size());
                });
            },
        );
    }

    let same_dtype_size = 1_000_000;
    let same_dtype_source = make_array(
        &backend,
        Shape::new(vec![same_dtype_size]),
        DType::Float64,
        MemoryOrder::C,
    );
    group.bench_with_input(
        BenchmarkId::new("astype_float64_to_float64", same_dtype_size),
        &same_dtype_size,
        |b, _| {
            b.iter(|| {
                let out = backend
                    .astype(black_box(&same_dtype_source), DType::Float64)
                    .expect("same-dtype astype should succeed");
                black_box(out.size());
            });
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_to_dmatrix_ab,
    bench_creation,
    bench_broadcast,
    bench_indexing,
    bench_dtype_cast
);
criterion_main!(benches);
