#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrationSeam {
    pub consumer_crate: &'static str,
    pub seam_name: &'static str,
    pub required_modules: &'static [&'static str],
    pub mode_requirements: &'static str,
}

pub const INTEGRATION_SEAMS: &[IntegrationSeam] = &[
    IntegrationSeam {
        consumer_crate: "fsci-linalg",
        seam_name: "input normalization and dtype promotion before solve/decompose",
        required_modules: &["creation", "dtype", "broadcast", "indexing"],
        mode_requirements: "Strict mode must preserve SciPy-observable semantics; hardened mode adds fail-closed shape and non-finite guards.",
    },
    IntegrationSeam {
        consumer_crate: "fsci-opt",
        seam_name: "objective argument coercion and broadcast-safe gradient evaluation",
        required_modules: &["dtype", "broadcast", "indexing"],
        mode_requirements: "Hardened mode rejects malformed objective metadata before iterative loops.",
    },
    IntegrationSeam {
        consumer_crate: "fsci-sparse",
        seam_name: "dense/sparse boundary conversion policy and copy/view guarantees",
        required_modules: &["creation", "dtype", "indexing"],
        mode_requirements: "Strict mode follows SciPy copy semantics; hardened mode blocks ambiguous aliasing paths.",
    },
];

pub const NALGEBRA_DMATRIX_INTEGRATION_POINTS: &[&str] = &[
    "fsci-linalg: convert contiguous 2D Float64 arrays into nalgebra::DMatrix<f64> without semantic drift.",
    "fsci-opt: map Jacobian/Hessian workspace buffers to nalgebra::DMatrix while preserving broadcasted parameter shapes.",
    "fsci-runtime diagnostics: retain shape/dtype metadata so DMatrix conversions can emit deterministic audit traces.",
];
