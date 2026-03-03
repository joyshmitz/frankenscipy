use crate::backend::ArrayApiBackend;
use crate::error::{ArrayApiError, ArrayApiErrorKind, ArrayApiResult};
use crate::types::{DType, MemoryOrder, ScalarValue, Shape};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreationRequest {
    pub shape: Shape,
    pub dtype: DType,
    pub order: MemoryOrder,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FullRequest {
    pub fill_value: ScalarValue,
    pub dtype: DType,
    pub order: MemoryOrder,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArangeRequest {
    pub start: ScalarValue,
    pub stop: ScalarValue,
    pub step: ScalarValue,
    pub dtype: Option<DType>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinspaceRequest {
    pub start: ScalarValue,
    pub stop: ScalarValue,
    pub num: usize,
    pub endpoint: bool,
    pub dtype: Option<DType>,
}

fn validate_shape(shape: &Shape) -> ArrayApiResult<()> {
    if shape.element_count().is_none() {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::Overflow,
            "shape element count overflow",
        ));
    }
    Ok(())
}

pub fn zeros<B: ArrayApiBackend>(
    backend: &B,
    request: &CreationRequest,
) -> ArrayApiResult<B::Array> {
    validate_shape(&request.shape)?;
    backend.zeros(&request.shape, request.dtype, request.order)
}

pub fn ones<B: ArrayApiBackend>(
    backend: &B,
    request: &CreationRequest,
) -> ArrayApiResult<B::Array> {
    validate_shape(&request.shape)?;
    backend.ones(&request.shape, request.dtype, request.order)
}

pub fn empty<B: ArrayApiBackend>(
    backend: &B,
    request: &CreationRequest,
) -> ArrayApiResult<B::Array> {
    validate_shape(&request.shape)?;
    backend.empty(&request.shape, request.dtype, request.order)
}

pub fn full<B: ArrayApiBackend>(
    backend: &B,
    shape: &Shape,
    request: &FullRequest,
) -> ArrayApiResult<B::Array> {
    validate_shape(shape)?;
    backend.full(shape, request.fill_value, request.dtype, request.order)
}

pub fn arange<B: ArrayApiBackend>(
    backend: &B,
    request: &ArangeRequest,
) -> ArrayApiResult<B::Array> {
    if request.step == ScalarValue::I64(0)
        || request.step == ScalarValue::U64(0)
        || request.step == ScalarValue::F64(0.0)
        || request.step == (ScalarValue::ComplexF64 { re: 0.0, im: 0.0 })
    {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::InvalidStep,
            "step must be nonzero",
        ));
    }
    backend.arange(request.start, request.stop, request.step, request.dtype)
}

pub fn linspace<B: ArrayApiBackend>(
    backend: &B,
    request: &LinspaceRequest,
) -> ArrayApiResult<B::Array> {
    backend.linspace(
        request.start,
        request.stop,
        request.num,
        request.endpoint,
        request.dtype,
    )
}

pub fn from_slice<B: ArrayApiBackend>(
    backend: &B,
    values: &[ScalarValue],
    request: &CreationRequest,
) -> ArrayApiResult<B::Array> {
    validate_shape(&request.shape)?;
    backend.array_from_slice(values, &request.shape, request.dtype, request.order)
}
