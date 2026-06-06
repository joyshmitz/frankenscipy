# Reject: Tiled-4 Rank-k Far Update Stage 4f

Bead: `frankenscipy-z65tz`

## Profile Target

The measured residual for the blocked-bidiagonal path remains the private
`1024x512` Golub-Kahan reduction and its far trailing update. Stage 4e showed
that scalar DLABRD panel recurrence is the wrong route; this trial tested a
small column-tiled variant of the already-kept fused far-update kernel.

## Baseline

Committed Stage 4d fused helper baseline:

- shape: `1024x512`
- panel start: `16,16`
- `k_count=16`
- scalar reference: `18.463455 ms`
- current fused helper: `6.830909 ms`
- digest: `0xd60df77cdefac734`
- scalar-to-fused speedup: `2.702928x`

## Lever

Trial private `apply_bidiag_fused_rank_k_update_tiled4`. It processes four
trailing columns per row so each row-panel load feeds four output cells. For
every cell, the arithmetic order is unchanged:

1. Iterate `k = 0..k_count` in ascending order.
2. Subtract `v_by_k_row[k,row] * y_by_col_k[col,k]`.
3. Subtract `x_by_k_row[k,row] * u_by_col_k[col,k]`.

The helper stayed private and unwired during the trial, so public `svd`,
`svdvals`, `lstsq`, `pinv`, rank/rcond thresholds, certificates, error classes,
ordering, tie-breaking, and RNG behavior remained unchanged.

## Behavior Proof

RCH `ts1`:

```text
cargo test -p fsci-linalg --release --lib bidiag_tiled_rank_k_update_matches_fused_bits --locked -- --nocapture
```

Result: passed. The proof covers multiple shapes, non-zero starts, non-multiple
of four column tails, and odd `k_count`; it compares every matrix cell by
`f64::to_bits` against the current fused helper.

## Rebench

RCH `vmi1264463` same-binary A/B:

- current fused helper: `28.084901 ms`
- tiled-4 helper: `29.051494 ms`
- fused digest: `0xd60df77cdefac734`
- tiled digest: `0xd60df77cdefac734`
- speedup: `0.966728x`

## Decision

Reject. Score: `0.0`: the fresh same-binary rebench did not clear the
Score >= 2.0 keep gate and was slightly slower than the current fused helper.

The source helper, bit-proof test, and ignored perf probe were restored out
before staging. The next pass should not retry column-loop micro-tiling; attack
reusable packed panel buffers feeding the kept fused far-update kernel, or a
two-stage communication-avoiding bidiagonalization path. Target for that next
pass remains at least `2.5x` over Stage 4a with reconstruction proof and public
golden-output guards before wiring.

## Validation

Passed during the trial:

- `cargo fmt -p fsci-linalg --check`
- RCH `ts1` bitwise equivalence test
- RCH `vmi1264463` same-binary perf probe

Post-rejection validation should run on the restored source before the rejection
record is committed.
