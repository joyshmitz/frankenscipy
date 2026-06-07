# Compact-WY Composition Rejection

Bead: `frankenscipy-7nlmr`

## Target

Continue after exact-order replay schedule changes failed. This pass tried the
first genuinely different block-reflector primitive from Alien Graveyard `9.6
Communication-Avoiding Algorithms`: compose a panel of Householder reflectors
and apply the panel as `C := C - Y*T*(Y^T*C)`.

Fresh baseline:

- Worker: `vmi1149989`
- Probe: `thin_bidiag_factor_replay_perf_probe`
- Shape: `1024x512`
- Dense-product reference: `603.920612 ms`
- Current serial reflector replay: `298.884297 ms`
- Replay digest: `0x8f521a39638fb520`

## Lever Tried

True compact-WY-style left reflector panel composition with panel size `8`.
The candidate built local `Y` columns and triangular `T` for each panel in the
left replay order, then applied the panel to the thin `U` factor with two
matrix-shaped contractions.

This intentionally changed floating-point grouping while preserving the same
Householder reflectors, singular values, right reflector replay, sign
canonicalization, rank/rcond thresholds, error behavior, and RNG absence.

## Proof

RCH `ts1` tolerance/reconstruction proof passed:

- `thin_bidiag_compact_wy_replay_matches_serial_tolerance`: passed
- Singular values stayed bit-identical
- Max allowed drift: `U <= 1e-10`, `Vt <= 1e-12`
- Reconstruction and orthogonality stayed within the existing thin-SVD
  tolerance budget

The same-binary perf probe also reported:

- `u_max_abs_diff=4.44089209850062616e-15`
- `vt_max_abs_diff=0.00000000000000000e0`
- Serial digest: `0x8f521a39638fb520`
- Compact-WY digest: `0x9efa22810dba3443`

## Rebench

Same-binary A/B on RCH `ts1`:

- Serial left replay: `245.225881 ms`
- Compact-WY panel replay: `386.421598 ms`
- Speedup: `0.634607x`

## Decision

Rejected. The prototype proved numerically acceptable against the tolerance
budget, but it was `1.58x` slower than serial replay and changed the exact thin
SVD digest, so it fails both the performance gate and the no-unapproved-golden
migration rule. Source was restored; no production code from this trial remains.

Score: `0.0`.

Next primitive: avoid dense per-panel `Y/T` allocation and naive contractions.
The next attack must either use a lower-overhead in-place block reflector
representation with reusable panel buffers, or pivot back to a two-stage
communication-avoiding bidiagonal reducer where block composition amortizes
trailing updates rather than just thin-factor replay.
