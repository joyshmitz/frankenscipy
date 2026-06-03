# Primitive Selection: PSD Welch indexed inner loop

## Target

- Bead: `frankenscipy-skjpy`
- Benchmark: `time_series/psd_welch/4096_w128_o64`
- RCH baseline: `[541.86 us, 549.72 us, 558.78 us]`
- Golden before: `85048a3c06ab045815cbeb238fee9e1e07a05c27ceed3c3782ec0fd5ea97c6b1`

## Profile Backing

The kept post-twiddle-SoA broad stats reprofile showed PSD Welch as the top
single stats row after QMC SoA rejection:
`time_series/psd_welch/4096_w128_o64` at
`[255.46 us, 257.41 us, 259.60 us]`. This bead captured a fresh focused RCH
baseline on current HEAD before any source edit.

## Selected Lever

Replace the innermost `segment.iter().zip(cos_row.iter()).zip(sin_row.iter())`
walk with an indexed `while` loop over the same three slices.

## Alien Primitive

This is a narrow loop-shape/cache primitive: keep the SoA twiddle layout but
make the hot numeric stream explicit and bounds-check-friendly for the optimizer.
It does not change traversal order or introduce unsafe code/SIMD.

## Isomorphism Contract

- Public validation and early returns stay unchanged.
- Hann window generation stays unchanged.
- Twiddle generation order stays frequency-major then sample-major.
- Segment order stays ascending `start`.
- Frequency order stays ascending `k`.
- Sample order stays ascending index within the segment.
- Output order stays ascending PSD frequency bin.
- Per-sample floating-point sequence remains `re += s * cos`, then `im -= s * sin`.
- RNG, tie-breaking, and global mutable state are absent except the existing
  `OnceLock` 128-point plan cache.

## Score

Target score: `3.0 = impact 2 * confidence 3 / effort 2`.

Keep only if PSD golden output is byte-identical after the change and focused
RCH Criterion shows a real win over the 549.72 us median baseline.
