use crate::backend::ArrayApiBackend;
use crate::error::ArrayApiResult;
use crate::types::{DType, ExecutionMode};

pub fn default_float_dtype(mode: ExecutionMode) -> DType {
    match mode {
        ExecutionMode::Strict => DType::Float64,
        ExecutionMode::Hardened => DType::Float64,
    }
}

pub fn result_type<B: ArrayApiBackend>(
    backend: &B,
    dtypes: &[DType],
    force_floating: bool,
) -> ArrayApiResult<DType> {
    backend.result_type(dtypes, force_floating)
}
