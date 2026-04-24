use crate::backend::ArrayApiBackend;
use crate::error::{ArrayApiError, ArrayApiErrorKind, ArrayApiResult};
use crate::types::{IndexExpr, Shape};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexingMode {
    Basic,
    Advanced,
    BooleanMask,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexRequest {
    pub mode: IndexingMode,
    pub index: IndexExpr,
}

pub fn getitem<B: ArrayApiBackend>(
    backend: &B,
    array: &B::Array,
    request: &IndexRequest,
) -> ArrayApiResult<B::Array> {
    match (&request.mode, &request.index) {
        (IndexingMode::Basic, IndexExpr::Basic { .. })
        | (IndexingMode::Advanced, IndexExpr::Advanced { .. })
        | (IndexingMode::BooleanMask, IndexExpr::BooleanMask { .. }) => {
            backend.getitem(array, &request.index)
        }
        _ => Err(ArrayApiError::new(
            ArrayApiErrorKind::InvalidIndex,
            "index expression does not match requested indexing mode",
        )),
    }
}

pub fn getitem_with_audit<B: ArrayApiBackend>(
    backend: &B,
    array: &B::Array,
    request: &IndexRequest,
    ledger: &crate::audit::SyncSharedAuditLedger,
) -> ArrayApiResult<B::Array> {
    let result = getitem(backend, array, request);
    if let Err(err) = &result {
        crate::audit::record_array_api_error(
            ledger,
            "getitem",
            format!("shape={:?}; request={request:?}", backend.shape_of(array)).as_bytes(),
            err.kind,
        );
    }
    result
}

pub fn reshape<B: ArrayApiBackend>(
    backend: &B,
    array: &B::Array,
    new_shape: &Shape,
) -> ArrayApiResult<B::Array> {
    backend.reshape(array, new_shape)
}

pub fn reshape_with_audit<B: ArrayApiBackend>(
    backend: &B,
    array: &B::Array,
    new_shape: &Shape,
    ledger: &crate::audit::SyncSharedAuditLedger,
) -> ArrayApiResult<B::Array> {
    let result = reshape(backend, array, new_shape);
    if let Err(err) = &result {
        crate::audit::record_array_api_error(
            ledger,
            "reshape",
            format!("from={:?}; to={new_shape:?}", backend.shape_of(array)).as_bytes(),
            err.kind,
        );
    }
    result
}

pub fn transpose<B: ArrayApiBackend>(backend: &B, array: &B::Array) -> ArrayApiResult<B::Array> {
    backend.transpose(array)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{ArrayApiBackend, CoreArrayBackend};
    use crate::types::{DType, ExecutionMode, MemoryOrder, ScalarValue, SliceSpec};

    fn strict_backend() -> CoreArrayBackend {
        CoreArrayBackend::new(ExecutionMode::Strict)
    }

    #[test]
    fn getitem_rejects_mode_index_mismatches() {
        let backend = strict_backend();
        let array = backend
            .array_from_slice(
                &[
                    ScalarValue::F64(0.0),
                    ScalarValue::F64(1.0),
                    ScalarValue::F64(2.0),
                ],
                &Shape::new(vec![3]),
                DType::Float64,
                MemoryOrder::C,
            )
            .expect("array should build");

        let request = IndexRequest {
            mode: IndexingMode::Basic,
            index: IndexExpr::Advanced {
                indices: vec![vec![0]],
            },
        };
        let err = getitem(&backend, &array, &request).expect_err("mode mismatch must fail");
        assert_eq!(err.kind, ArrayApiErrorKind::InvalidIndex);
    }

    #[test]
    fn getitem_reshape_and_transpose_cover_success_paths() {
        let backend = strict_backend();
        let array = backend
            .array_from_slice(
                &[
                    ScalarValue::F64(1.0),
                    ScalarValue::F64(2.0),
                    ScalarValue::F64(3.0),
                    ScalarValue::F64(4.0),
                    ScalarValue::F64(5.0),
                    ScalarValue::F64(6.0),
                ],
                &Shape::new(vec![2, 3]),
                DType::Float64,
                MemoryOrder::C,
            )
            .expect("array should build");

        let request = IndexRequest {
            mode: IndexingMode::Basic,
            index: IndexExpr::Basic {
                slices: vec![
                    SliceSpec {
                        start: Some(0),
                        stop: Some(2),
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
        let sliced = getitem(&backend, &array, &request).expect("basic indexing should succeed");
        assert_eq!(sliced.shape(), &Shape::new(vec![2, 2]));

        let transposed = transpose(&backend, &array).expect("transpose should succeed");
        assert_eq!(transposed.shape(), &Shape::new(vec![3, 2]));

        let reshaped = reshape(&backend, &array, &Shape::new(vec![3, 2]))
            .expect("reshape with equal element count should succeed");
        assert_eq!(reshaped.shape(), &Shape::new(vec![3, 2]));
    }

    #[test]
    fn getitem_boolean_mask_and_transpose_error_paths_are_reported() {
        let backend = strict_backend();
        let one_d = backend
            .array_from_slice(
                &[
                    ScalarValue::F64(1.0),
                    ScalarValue::F64(2.0),
                    ScalarValue::F64(3.0),
                ],
                &Shape::new(vec![3]),
                DType::Float64,
                MemoryOrder::C,
            )
            .expect("array should build");

        let matching_mask = IndexRequest {
            mode: IndexingMode::BooleanMask,
            index: IndexExpr::BooleanMask {
                mask_shape: Shape::new(vec![3]),
            },
        };
        let selected =
            getitem(&backend, &one_d, &matching_mask).expect("matching mask shape should pass");
        assert_eq!(selected.shape(), &Shape::new(vec![3]));

        let mismatch_mask = IndexRequest {
            mode: IndexingMode::BooleanMask,
            index: IndexExpr::BooleanMask {
                mask_shape: Shape::new(vec![2]),
            },
        };
        let err =
            getitem(&backend, &one_d, &mismatch_mask).expect_err("mask shape mismatch should fail");
        assert_eq!(err.kind, ArrayApiErrorKind::InvalidShape);

        let transpose_err = transpose(&backend, &one_d).expect_err("rank-1 transpose should fail");
        assert_eq!(transpose_err.kind, ArrayApiErrorKind::InvalidShape);
    }
}
