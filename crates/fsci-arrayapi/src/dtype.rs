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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CoreArrayBackend;

    #[test]
    fn default_float_dtype_is_float64_in_both_modes() {
        assert_eq!(default_float_dtype(ExecutionMode::Strict), DType::Float64);
        assert_eq!(default_float_dtype(ExecutionMode::Hardened), DType::Float64);
    }

    #[test]
    fn result_type_delegates_to_backend_rules() {
        let backend = CoreArrayBackend::new(ExecutionMode::Strict);
        assert_eq!(
            result_type(&backend, &[DType::Float32, DType::Complex64], false)
                .expect("promotion should succeed"),
            DType::Complex64
        );
        assert_eq!(
            result_type(&backend, &[], false).expect("empty list should use default float dtype"),
            DType::Float64
        );
        assert_eq!(
            result_type(&backend, &[DType::Int32], true)
                .expect("force_floating should permit integral inputs"),
            DType::Float64
        );
        assert_eq!(
            result_type(&backend, &[DType::Int32], false)
                .expect("integral dtype promotion should be supported"),
            DType::Int32
        );
        assert_eq!(
            result_type(&backend, &[DType::Bool, DType::UInt16], false)
                .expect("bool should promote to the numeric dtype"),
            DType::UInt16
        );
    }
}
