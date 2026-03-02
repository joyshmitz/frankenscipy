#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::types::{
    DispatchPlan, DispatchStep, KernelRegime, SpecialResult, SpecialTensor, not_yet_implemented,
};

pub const HYPER_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "hyp1f1",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|z| <= 2 and moderate parameters",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "parameter shifting to stable region",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z| or large parameters",
            },
        ],
        notes: "Fallback routing should preserve SciPy branch-selection semantics for strict mode.",
    },
    DispatchPlan {
        function: "hyp2f1",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|z| < 0.9 and c not near nonpositive integers",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "boundary neighborhoods near z=1",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "contiguous relation stabilization",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large-parameter asymptotic domains",
            },
        ],
        notes: "z=1 convergence edge cases and c-pole exclusions are explicit hardened guards.",
    },
];

pub fn hyp1f1(
    _a: &SpecialTensor,
    _b: &SpecialTensor,
    _z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    not_yet_implemented("hyp1f1", mode, "P2C-006-D skeleton only")
}

pub fn hyp2f1(
    _a: &SpecialTensor,
    _b: &SpecialTensor,
    _c: &SpecialTensor,
    _z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    not_yet_implemented("hyp2f1", mode, "P2C-006-D skeleton only")
}
