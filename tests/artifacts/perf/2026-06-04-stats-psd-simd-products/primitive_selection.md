# PSD Welch SIMD Product-Lane Primitive

Bead: `frankenscipy-65gu6`

## Profile Target

Fresh crate-scoped RCH Criterion reprofile:

- `time_series/psd_welch/4096_w128_o64`: `382.47 us` median, top stats row.

Focused RCH Criterion baseline before source edits:

- `time_series/psd_welch/4096_w128_o64`: `[573.01 us, 574.37 us, 575.49 us]`.

Golden SHA-256 before source edits:

`85048a3c06ab045815cbeb238fee9e1e07a05c27ceed3c3782ec0fd5ea97c6b1`

## Prior PSD Shapes Excluded

- Segment-buffer reuse: rejected.
- Stack segment: rejected.
- Frequency-major segment traversal: rejected.
- Indexed scalar inner loop: rejected.
- Twiddle SoA and 128-point plan cache: already kept; do not repeat.
- FFT or reordered reductions: excluded because they change floating-point
  summation order and cannot satisfy the bitwise golden contract.

## Selected Primitive

Use safe Rust portable SIMD for the multiplication half of the DFT inner loop:
compute `s * cos` and `s * sin` in fixed-width lanes, then replay the scalar
`re +=` and `im -=` additions lane by lane in the original sample order.

This applies the no-gaps safe-SIMD primitive to the profiled data-plane kernel
without using unsafe code or external BLAS/FFT backends.

## Isomorphism Obligations

- Validation and early-return order stay unchanged.
- Hann window and twiddle generation stay unchanged.
- Segment order stays ascending.
- Frequency order stays ascending.
- Sample order inside each segment-frequency DFT stays ascending.
- Floating-point products remain exactly `s * cos` and `s * sin`.
- Floating-point additions remain scalar and ordered: `re += product`, then
  `im -= product` for each sample lane in ascending sample order.
- Output PSD bin order and final averaging stay unchanged.
- RNG, tie-breaking, and global mutable numerical state are absent.

## Score Target

`4.0 = impact 2 * confidence 4 / effort 2`; reject if the golden SHA changes or
if focused RCH Criterion does not show a real win.
