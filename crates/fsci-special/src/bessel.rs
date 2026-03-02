#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::types::{
    DispatchPlan, DispatchStep, KernelRegime, SpecialResult, SpecialTensor, not_yet_implemented,
};

pub const BESSEL_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "jv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small |z|",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "integer-order and moderate |z| windows",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z| oscillatory region",
            },
        ],
        notes: "Negative-order reconstruction uses SciPy-compatible cancellation-safe branches.",
    },
    DispatchPlan {
        function: "yv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small |z| away from singular neighborhoods",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "moderate argument / order coupling",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z|",
            },
        ],
        notes: "Origin-adjacent singularity behavior must preserve SciPy divergence semantics.",
    },
    DispatchPlan {
        function: "iv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small |z|",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "order shifting in stable window",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z| or large order",
            },
        ],
        notes: "Real-negative argument handling enforces strict integer-order branch rules.",
    },
    DispatchPlan {
        function: "kv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small positive |z|",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large positive real z",
            },
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "complex continuation branch",
            },
        ],
        notes: "Hardened mode adds singular-neighborhood diagnostics near zero.",
    },
    DispatchPlan {
        function: "hankel1",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "AMOS-compatible principal branch composition",
            },
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "negative-order phase mapping",
            },
        ],
        notes: "Outgoing-wave sign and phase conventions are contract-critical.",
    },
    DispatchPlan {
        function: "hankel2",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "AMOS-compatible principal branch composition",
            },
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "negative-order phase mapping",
            },
        ],
        notes: "Incoming-wave sign and phase conventions are contract-critical.",
    },
];

pub fn jv(_v: &SpecialTensor, _z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("jv", mode, "P2C-006-D skeleton only")
}

pub fn yv(_v: &SpecialTensor, _z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("yv", mode, "P2C-006-D skeleton only")
}

pub fn iv(_v: &SpecialTensor, _z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("iv", mode, "P2C-006-D skeleton only")
}

pub fn kv(_v: &SpecialTensor, _z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("kv", mode, "P2C-006-D skeleton only")
}

pub fn hankel1(_v: &SpecialTensor, _z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("hankel1", mode, "P2C-006-D skeleton only")
}

pub fn hankel2(_v: &SpecialTensor, _z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    not_yet_implemented("hankel2", mode, "P2C-006-D skeleton only")
}
