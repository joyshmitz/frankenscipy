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

pub fn reshape<B: ArrayApiBackend>(
    backend: &B,
    array: &B::Array,
    new_shape: &Shape,
) -> ArrayApiResult<B::Array> {
    backend.reshape(array, new_shape)
}

pub fn transpose<B: ArrayApiBackend>(backend: &B, array: &B::Array) -> ArrayApiResult<B::Array> {
    backend.transpose(array)
}
