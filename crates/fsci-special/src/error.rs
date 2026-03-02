#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::types::{
    DispatchPlan, DispatchStep, KernelRegime, SpecialResult, SpecialTensor, not_yet_implemented,
};

pub const ERROR_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "erf",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|z| < 1",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "|z| >= 1",
            },
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "complex continuation region",
            },
        ],
        notes: "Strict mode preserves endpoint parity: erf(0)=0, erf(+/-inf)=+/-1.",
    },
    DispatchPlan {
        function: "erfc",
        steps: &[DispatchStep {
            regime: KernelRegime::BackendDelegate,
            when: "use dedicated complement kernel to avoid 1-erf cancellation",
        }],
        notes: "Central requirement: erf(x)+erfc(x)=1 within tolerance on finite reals.",
    },
    DispatchPlan {
        function: "erfinv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|y| < 0.9",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "0.9 <= |y| < 1",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "polish via Newton/Halley refinement",
            },
        ],
        notes: "Endpoints y=+/-1 map to +/-inf with strict SciPy parity.",
    },
    DispatchPlan {
        function: "erfcinv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "map to erfinv(1-y) with tail-stable correction",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "optional refinement iteration",
            },
        ],
        notes: "Domain is [0,2] in strict mode with hardened fail-closed diagnostics for malformed inputs.",
    },
];

pub fn erf(_z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("erf", mode, "P2C-006-D skeleton only")
}

pub fn erfc(_z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("erfc", mode, "P2C-006-D skeleton only")
}

pub fn erfinv(_y: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("erfinv", mode, "P2C-006-D skeleton only")
}

pub fn erfcinv(_y: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("erfcinv", mode, "P2C-006-D skeleton only")
}
