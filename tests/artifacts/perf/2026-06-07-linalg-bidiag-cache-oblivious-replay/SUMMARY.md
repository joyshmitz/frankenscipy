# Bidiag Replay Column-Slice Rejection

Bead: `frankenscipy-8l8r1.46`

## Target

Continue the deeper bidiagonal SVD replay route after the right-workspace
allocation lever failed. The fresh RCH baseline was:

- Worker: `vmi1156319`
- Probe: `thin_bidiag_factor_replay_perf_probe`
- Shape: `1024x512`
- Dense-product reference: `1148.421774 ms`
- Reflector replay: `426.149221 ms`
- Replay digest: `0x8f521a39638fb520`

## Lever Tried

Replace the left Householder replay helper's indexed `DMatrix[(row, col)]`
accesses with direct column-major slice streaming over each affected column.

This preserved reflector order, per-column dot-product order, update order,
right-reflector replay, sign canonicalization, singular-value ordering, rank
threshold behavior, error behavior, and RNG absence.

## Proof

RCH `ts1` focused replay proof passed:

- `thin_bidiag_reflector_replay_matches_dense_product_reference`: passed

RCH `ts1` same-binary bit proof passed during the trial:

- `thin_bidiag_column_slice_replay_matches_indexed_reference_bits`: passed
- Equality policy: `U` and `Vt` entries compared by `f64::to_bits`
- Digest: unchanged at `0x8f521a39638fb520`

## Rebench

Same-binary A/B on RCH `ts1`:

- Indexed left replay: `259.903851 ms`
- Column-slice left replay: `262.186587 ms`
- Speedup: `0.991293x`
- Digest: `0x8f521a39638fb520`

## Decision

Rejected. The candidate preserved bits but regressed by about `0.9%`, so it does
not clear the Score `>= 2.0` keep gate. Source was restored; no production code
from this trial remains.

Score: `0.0`.

Next primitive: leave index/access cleanup behind and attack a structurally
different communication-avoiding path: either packed compact-WY/block reflector
replay with one GEMM-shaped application per panel, or a two-stage tiled
bidiagonal reducer that amortizes far trailing updates with reusable safe-Rust
panel buffers.

## Addendum: Transposed Right-Replay Layout Trial

An additional no-ship probe tried replaying right Householder reflectors over
`Vt^T` with the existing left-Householder kernel, then transposing back. For each
original row, the direct-right replay dot/update offset order was preserved.

Proof:

- RCH `ts1` bit proof passed:
  `thin_bidiag_transposed_right_replay_matches_direct_right_bits`
- Public SVD/lstsq/pinv SHA stayed
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`
- Replay digest stayed `0x8f521a39638fb520`

Rebench on RCH `ts1`:

```text
baseline replay_ms=250.248466
after    replay_ms=278.751183
```

Decision: rejected. The transpose cost and layout churn outweighed locality
improvement; source was restored.
