# frankenscipy-8l8r1.87 compact-WY panel contract

## Primitive

Add only a private compact-WY symmetric panel update probe. The slice builds a
panel representation `Q = I - V T V^T` from a small group of Householder
reflectors, then applies the symmetric trailing update:

```text
Y = A V T
W = Y - 1/2 V T^T V^T Y
A' = A - V W^T - W V^T
```

This is the BLAS-3-style far-update kernel that the rejected scalar-panel
full-to-band route was missing. It is not a public `eigh` route and must not
change public validation, output ordering, trace behavior, or nalgebra fallback.

## One-lever boundary

- Add private `CompactWyPanel` state and helpers near the existing
  `HouseholderReflector` utilities in `crates/fsci-linalg/src/lib.rs`.
- Add focused unit proof and ignored release perf probe in the existing linalg
  test module.
- Do not wire public `eigh`, `eigvalsh`, `eig_banded`, or benchmark harness
  dispatch in this slice.
- Do not repeat the `.86` scalar-panel full-to-band reducer.

## Proof obligations

- Build deterministic embedded Householder reflectors with no RNG.
- Prove compact-WY update matches scalar reflector replay for the same
  reflectors and active symmetric block.
- Prove symmetry drift, max absolute matrix difference, and public `eigh`
  golden digest stability.
- Preserve no-unsafe and no external BLAS/LAPACK/MKL/XLA.

## Performance gate

Compare compact-WY panel update against scalar reflector replay on the same
deterministic `512x512` active block through RCH release tests.

Keep only if:

- same-worker RCH runs pass,
- compact-WY is materially faster on the `512x512` panel update,
- Score `(Impact * Confidence) / Effort >= 2.0`,
- source remains private and proof-clean.

If the kernel fails, restore source and route deeper to band-to-tridiagonal
bulge chasing or a more cache-oblivious panel layout. Do not report a ceiling.
