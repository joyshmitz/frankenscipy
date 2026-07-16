use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_arrayapi::{
    ArrayApiArray, ArrayApiBackend, ArrayApiResult, CoreArray, CoreArrayBackend, CreationRequest,
    DType, ExecutionMode, FullRequest, IndexExpr, IndexRequest, IndexingMode, MemoryOrder,
    ScalarValue, Shape, SliceSpec, broadcast_shapes, from_slice, full, getitem,
    promote_and_broadcast, zeros,
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

fn row_major_strides_original(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; dims.len()];
    let mut stride = 1usize;
    for pos in (0..dims.len()).rev() {
        strides[pos] = stride;
        stride = stride.saturating_mul(dims[pos].max(1));
    }
    strides
}

fn advance_row_major_coords_original(coords: &mut [usize], dims: &[usize]) {
    for axis in (0..coords.len()).rev() {
        coords[axis] += 1;
        if coords[axis] < dims[axis] {
            break;
        }
        coords[axis] = 0;
    }
}

fn rank3_middle_singleton_broadcast_original(
    array: &CoreArray,
    shape: &Shape,
) -> (Shape, DType, Vec<ScalarValue>, MemoryOrder) {
    let target_shape = broadcast_shapes(&[array.shape().clone(), shape.clone()])
        .expect("benchmark shapes should broadcast");
    assert_eq!(target_shape, *shape);

    let output_size = shape
        .element_count()
        .expect("benchmark target shape should not overflow");
    let out_rank = shape.rank();
    let in_rank = array.shape().rank();
    let in_strides = row_major_strides_original(&array.shape().dims);
    let mut out_coords = vec![0usize; out_rank];
    let mut values = Vec::with_capacity(output_size);
    for linear in 0..output_size {
        let mut in_linear = 0usize;
        for (in_dim_idx, in_dim) in array.shape().dims.iter().enumerate() {
            let out_dim_idx = out_rank - in_rank + in_dim_idx;
            let coord = if *in_dim == 1 {
                0
            } else {
                out_coords[out_dim_idx]
            };
            in_linear += coord * in_strides[in_dim_idx];
        }
        values.push(array.values()[in_linear]);
        if linear + 1 < output_size {
            advance_row_major_coords_original(&mut out_coords, &shape.dims);
        }
    }

    (shape.clone(), array.dtype(), values, array.order())
}

fn bench_rank3_middle_singleton_broadcast_ab(c: &mut Criterion) {
    let backend = strict_backend();
    let array = make_array(
        &backend,
        Shape::new(vec![256, 1, 64]),
        DType::Float64,
        MemoryOrder::C,
    );
    let target_shape = Shape::new(vec![256, 16, 64]);
    let candidate = backend
        .broadcast_to(&array, &target_shape)
        .expect("rank-3 middle-singleton broadcast should succeed");
    let original = rank3_middle_singleton_broadcast_original(&array, &target_shape);
    assert_eq!(candidate.shape(), &original.0);
    assert_eq!(candidate.dtype(), original.1);
    assert_eq!(candidate.values(), original.2.as_slice());
    assert_eq!(candidate.order(), original.3);

    let mut group = c.benchmark_group("arrayapi_rank3_middle_singleton_broadcast_ab");
    group.bench_function("original_coordinate_walk/256x1x64_to_256x16x64", |b| {
        b.iter(|| {
            black_box(rank3_middle_singleton_broadcast_original(
                black_box(&array),
                black_box(&target_shape),
            ))
        });
    });
    group.bench_function("candidate_block_repeat/256x1x64_to_256x16x64", |b| {
        b.iter(|| {
            black_box(
                backend
                    .broadcast_to(black_box(&array), black_box(&target_shape))
                    .expect("rank-3 middle-singleton broadcast should succeed"),
            )
        });
    });
    group.finish();
}

fn promote_and_broadcast_identity_cast_original(
    backend: &CoreArrayBackend,
    arrays: &[&CoreArray],
    force_floating: bool,
) -> ArrayApiResult<Vec<CoreArray>> {
    let dtypes = arrays
        .iter()
        .map(|array| backend.dtype_of(array))
        .collect::<Vec<_>>();
    let target_dtype = backend.result_type(&dtypes, force_floating)?;
    let target_shape = broadcast_shapes(
        &arrays
            .iter()
            .map(|array| backend.shape_of(array))
            .collect::<Vec<_>>(),
    )?;
    arrays
        .iter()
        .map(|array| {
            let cast = backend.astype(array, target_dtype)?;
            backend.broadcast_to(&cast, &target_shape)
        })
        .collect()
}

fn bench_promote_identity_cast_ab(c: &mut Criterion) {
    let backend = strict_backend();
    let shape = Shape::new(vec![256, 256]);
    let left = make_array(&backend, shape.clone(), DType::Float64, MemoryOrder::C);
    let right = make_array(&backend, shape, DType::Float64, MemoryOrder::C);
    let current = promote_and_broadcast(&backend, &[&left, &right], false)
        .expect("optimized promotion should succeed");
    let original = promote_and_broadcast_identity_cast_original(&backend, &[&left, &right], false)
        .expect("original promotion should succeed");
    assert_eq!(current, original);

    let mut group = c.benchmark_group("arrayapi_promote_identity_cast_ab");
    group.bench_function("original_identity_cast/256x256x2", |b| {
        b.iter(|| {
            black_box(
                promote_and_broadcast_identity_cast_original(
                    black_box(&backend),
                    &[black_box(&left), black_box(&right)],
                    false,
                )
                .expect("original promotion should succeed"),
            )
        })
    });
    group.bench_function("candidate_dtype_guard/256x256x2", |b| {
        b.iter(|| {
            black_box(
                promote_and_broadcast(
                    black_box(&backend),
                    &[black_box(&left), black_box(&right)],
                    false,
                )
                .expect("optimized promotion should succeed"),
            )
        })
    });
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
    bench_rank3_middle_singleton_broadcast_ab,
    bench_promote_identity_cast_ab,
    bench_indexing,
    bench_dtype_cast
);
criterion_main!(benches);
