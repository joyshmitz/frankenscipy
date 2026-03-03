#![forbid(unsafe_code)]

pub mod backend;
pub mod broadcast;
pub mod creation;
pub mod dtype;
pub mod error;
pub mod indexing;
pub mod integration;
pub mod types;

pub use backend::{
    ArrayApiArray, ArrayApiBackend, CoreArray, CoreArrayBackend, DTypeDispatchLog, ShapeMismatchLog,
};
pub use broadcast::{broadcast_shapes, promote_and_broadcast};
pub use creation::{
    ArangeRequest, CreationRequest, FullRequest, LinspaceRequest, arange, empty, from_slice, full,
    linspace, ones, zeros,
};
pub use dtype::{default_float_dtype, result_type};
pub use error::{ArrayApiError, ArrayApiErrorKind, ArrayApiResult};
pub use indexing::{IndexRequest, IndexingMode, getitem, reshape, transpose};
pub use integration::{INTEGRATION_SEAMS, NALGEBRA_DMATRIX_INTEGRATION_POINTS};
pub use types::{DType, ExecutionMode, IndexExpr, MemoryOrder, ScalarValue, Shape, SliceSpec};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_boundaries_are_exported() {
        assert!(
            INTEGRATION_SEAMS
                .iter()
                .any(|seam| seam.consumer_crate == "fsci-linalg")
        );
        assert!(
            NALGEBRA_DMATRIX_INTEGRATION_POINTS
                .iter()
                .any(|entry| entry.contains("DMatrix"))
        );
    }
}
