# frankenscipy-psn7x.10 Baseline - Implicit Symmetric-Band Bulge Chase

## Target

- Bead: `frankenscipy-psn7x.10`
- Target crate: `fsci-linalg`
- Primitive route: true implicit symmetric-band bulge chase with accumulated
  orthogonal transforms, or a transformed live-basis compact-WY panel generator.

## Current-Head Baseline

RCH worker: `vmi1152480`

Command:

```text
cargo test -j 1 -p fsci-linalg --lib symmetric_eigh_native_stage_breakdown_probe --release --locked -- --ignored --nocapture
```

| shape | reduction | tridiagonal | backtransform | sort | max abs diff | values digest |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 400x400 | 17.936892 ms | 13.141569 ms | 17.878834 ms | 0.716169 ms | 5.24025267623073887e-13 | `0x0dbbde75b75c8612` |
| 800x800 | 198.522580 ms | 59.087364 ms | 247.620325 ms | 2.423605 ms | 1.71418435002124170e-12 | `0x4461962827bdb038` |
| 1200x1200 | 513.536928 ms | 97.829834 ms | 240.652864 ms | 10.078261 ms | 1.79412040779425297e-12 | `0x2fc45e1f18ceb0ab` |

## Contract Notes

- Ordering/tie behavior: the current route sorts eigenpairs through the existing
  deterministic comparator path; any candidate must preserve public ordering.
- RNG behavior: no public RNG is used by this probe or the candidate route.
- Floating-point contract: acceptance is tolerance-based against dense/native
  residuals, orthogonality, and golden digests because the proposed primitive is
  algorithmically different from the current dense route.
- Safety: safe Rust only; no C BLAS/LAPACK/MKL/XLA linkage.

## Decision

Productive baseline only. No production source was changed for this pass.
