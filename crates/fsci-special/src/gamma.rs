#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::types::{
    DispatchPlan, DispatchStep, KernelRegime, SpecialResult, SpecialTensor, not_yet_implemented,
};

pub const GAMMA_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "gamma",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "Re(z) < 0.5 and not at poles",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "0.5 <= Re(z) < 8.0",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "Re(z) >= 8.0",
            },
        ],
        notes: "Strict mode preserves SciPy signed-zero and pole semantics; hardened mode adds fail-closed diagnostics.",
    },
    DispatchPlan {
        function: "gammaln",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "real axis shift into stable window",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |x| Stirling region",
            },
        ],
        notes: "Keep +inf pole parity with SciPy and avoid direct exp(gammaln) overflow back-conversions.",
    },
    DispatchPlan {
        function: "digamma",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "Re(z) <= 0 and away from poles",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "shift toward asymptotic region",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z|",
            },
        ],
        notes: "Negative-axis cancellation regions require dedicated numerical guards before D2 implementation.",
    },
    DispatchPlan {
        function: "polygamma",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "order n fixed and argument shifted off poles",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large argument regime",
            },
        ],
        notes: "Polygamma order must remain integer and nonnegative in both modes.",
    },
    DispatchPlan {
        function: "rgamma",
        steps: &[DispatchStep {
            regime: KernelRegime::BackendDelegate,
            when: "evaluate via reciprocal-safe gamma path",
        }],
        notes: "Prefer reciprocal from stabilized gammaln/gamma path to reduce overflow risk.",
    },
];

pub fn gamma(_z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("gamma", mode, "P2C-006-D skeleton only")
}

pub fn gammaln(_x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("gammaln", mode, "P2C-006-D skeleton only")
}

pub fn digamma(_z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("digamma", mode, "P2C-006-D skeleton only")
}

pub fn polygamma(_n: usize, _z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("polygamma", mode, "P2C-006-D skeleton only")
}

pub fn rgamma(_z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("rgamma", mode, "P2C-006-D skeleton only")
}
