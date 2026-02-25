#![forbid(unsafe_code)]

//! FFT API surface for FrankenSciPy packet P2C-005.
//!
//! This crate is intentionally contract-first at this stage:
//! - module boundaries are fixed (`transforms`, `helpers`, `plan`)
//! - public signatures are stable for conformance wiring
//! - kernels are populated in subsequent packet beads

pub mod helpers;
pub mod plan;
pub mod transforms;

pub use helpers::{fftfreq, fftshift_1d, ifftshift_1d, rfftfreq};
pub use plan::{
    CacheAdmissionPolicy, PlanCacheBackend, PlanCacheConfig, PlanFingerprint, PlanKey,
    PlanMetadata, PlanningStrategy,
};
pub use transforms::{
    BackendKind, Complex64, FftError, FftOptions, TransformTrace, WorkerPolicy, fft, fft2, fftn,
    ifft, ifft2, irfft, rfft, take_transform_traces,
};

/// FFT normalization modes matching SciPy/PocketFFT conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum Normalization {
    Forward,
    #[default]
    Backward,
    Ortho,
}

/// Transform entrypoints represented in the packet boundary contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TransformKind {
    Fft,
    Ifft,
    Rfft,
    Irfft,
    Fft2,
    Ifft2,
    Fftn,
}

#[cfg(test)]
mod tests {
    use super::{Normalization, TransformKind};

    #[test]
    fn normalization_default_matches_scipy() {
        assert_eq!(Normalization::default(), Normalization::Backward);
    }

    #[test]
    fn transform_kind_order_is_stable_for_plan_keys() {
        assert!(TransformKind::Fft < TransformKind::Fftn);
    }
}
