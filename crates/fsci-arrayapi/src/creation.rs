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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{ArrayApiArray, CoreArrayBackend};
    use crate::types::ExecutionMode;

    fn strict_backend() -> CoreArrayBackend {
        CoreArrayBackend::new(ExecutionMode::Strict)
    }

    #[test]
    fn constructors_reject_overflowing_shape() {
        let backend = strict_backend();
        let request = CreationRequest {
            shape: Shape::new(vec![usize::MAX, 2]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let full_request = FullRequest {
            fill_value: ScalarValue::F64(1.0),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };

        for result in [
            zeros(&backend, &request),
            ones(&backend, &request),
            empty(&backend, &request),
            from_slice(&backend, &[ScalarValue::F64(1.0)], &request),
            full(&backend, &request.shape, &full_request),
        ] {
            let err = result.expect_err("overflowing shape must fail");
            assert_eq!(err.kind, ArrayApiErrorKind::Overflow);
        }
    }

    #[test]
    fn arange_rejects_zero_step_for_all_supported_scalar_variants() {
        let backend = strict_backend();
        for step in [
            ScalarValue::I64(0),
            ScalarValue::U64(0),
            ScalarValue::F64(0.0),
            ScalarValue::ComplexF64 { re: 0.0, im: 0.0 },
        ] {
            let request = ArangeRequest {
                start: ScalarValue::F64(0.0),
                stop: ScalarValue::F64(3.0),
                step,
                dtype: Some(DType::Float64),
            };
            let err = arange(&backend, &request).expect_err("zero step must fail");
            assert_eq!(err.kind, ArrayApiErrorKind::InvalidStep);
        }
    }

    #[test]
    fn linspace_and_full_generate_expected_shapes() {
        let backend = strict_backend();
        let linspace_request = LinspaceRequest {
            start: ScalarValue::F64(-1.0),
            stop: ScalarValue::F64(1.0),
            num: 5,
            endpoint: true,
            dtype: Some(DType::Float32),
        };
        let lin = linspace(&backend, &linspace_request).expect("linspace should succeed");
        assert_eq!(lin.shape(), &Shape::new(vec![5]));
        assert_eq!(lin.dtype(), DType::Float32);

        let full_request = FullRequest {
            fill_value: ScalarValue::ComplexF64 { re: 2.0, im: -3.0 },
            dtype: DType::Complex64,
            order: MemoryOrder::F,
        };
        let full_array =
            full(&backend, &Shape::new(vec![2, 2]), &full_request).expect("full should succeed");
        assert_eq!(full_array.shape(), &Shape::new(vec![2, 2]));
        assert_eq!(full_array.dtype(), DType::Complex64);
        assert_eq!(full_array.size(), 4);
    }
}
