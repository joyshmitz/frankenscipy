#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::types::{
    DispatchPlan, DispatchStep, KernelRegime, SpecialResult, SpecialTensor, not_yet_implemented,
};

pub const BETA_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "beta",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "use gamma/gammaln composition in stable space",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large-parameter regime uses logspace stabilization",
            },
        ],
        notes: "Symmetry beta(a,b)=beta(b,a) must hold in strict mode and hardened mode.",
    },
    DispatchPlan {
        function: "betaln",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "direct logspace composition",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "a+b sufficiently large",
            },
        ],
        notes: "Primary path for underflow-prone beta regions.",
    },
    DispatchPlan {
        function: "betainc",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "x in lower-tail region",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "x in upper-tail region",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "parameter shifts for stability",
            },
        ],
        notes: "Strict mode preserves SciPy endpoint behavior at x=0 and x=1.",
    },
];

pub fn beta(_a: &SpecialTensor, _b: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("beta", mode, "P2C-006-D skeleton only")
}

pub fn betaln(_a: &SpecialTensor, _b: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("betaln", mode, "P2C-006-D skeleton only")
}

pub fn betainc(
    _a: &SpecialTensor,
    _b: &SpecialTensor,
    _x: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    not_yet_implemented("betainc", mode, "P2C-006-D skeleton only")
}
