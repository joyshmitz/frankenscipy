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

    /// Per br-8ghr: the INTEGRATION_SEAMS declarations are
    /// *aspirational*. Until an entry is backed by a real Cargo
    /// dependency on fsci-arrayapi from the named consumer crate,
    /// it describes design intent only. This test documents today's
    /// live vs aspirational split so an agent adding a Cargo
    /// dependency has a place to flip the label.
    ///
    /// The test deliberately asserts current state (all aspirational)
    /// rather than required state; flip the expected set when a new
    /// consumer wires up.
    #[test]
    fn integration_seam_declarations_are_aspirational() {
        let declared: Vec<&str> =
            INTEGRATION_SEAMS.iter().map(|s| s.consumer_crate).collect();
        // Live consumers (have a Cargo dep on fsci-arrayapi):
        //   - fsci-conformance (not in INTEGRATION_SEAMS; it's the
        //     Array API test packet driver, not an integration seam).
        // Aspirational (in INTEGRATION_SEAMS but no Cargo dep yet):
        const ASPIRATIONAL: &[&str] = &["fsci-linalg", "fsci-opt", "fsci-sparse"];
        for name in ASPIRATIONAL {
            assert!(
                declared.contains(name),
                "{name} dropped from INTEGRATION_SEAMS but was expected to be aspirational"
            );
        }
    }
}
