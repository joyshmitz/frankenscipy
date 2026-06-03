# QMC 2D Discrepancy SoA Negative Result

Bead: `frankenscipy-rh43r`

## Verdict

Rejected. The SoA point-cache layout preserved the QMC golden output exactly
but regressed the focused RCH benchmark, so no source change is kept.

## Profile-Backed Target

Latest post-PSD stats reprofile kept QMC 2D discrepancy rows prominent, with
`qmc_discrepancy/mixture/512x2` selected as the focused target.

Fresh focused baseline:

- `qmc_discrepancy/mixture/512x2`: `[276.30 us, 282.08 us, 287.92 us]` on `vmi1227854`.

## One Lever Tested

Split the private 2D discrepancy cached point data from one AoS vector into
parallel `x0`, `x1`, `centered0`, `centered1`, `abs0`, and `abs1` arrays.

The rejected `delta.powi(2)` replacement was not repeated.

## Behavior Proof

QMC golden before and after the candidate matched byte-for-byte.

SHA256:

`1fb5885cc35367f57b0e818e165a28f87cbb0b9a43fdc7ba4728a6778af44daf`

Preserved surfaces:

- Validation order and error surfaces.
- Row-major sample order.
- Pair loop order: `i` outer, `j` inner.
- Coordinate order: 0 then 1.
- Formula term order, including `delta.powi(2)`.
- RNG absence, tie-breaking absence, and global-state absence.

## Benchmark Gate

Focused RCH after:

- `qmc_discrepancy/mixture/512x2`: `[291.53 us, 297.89 us, 304.31 us]` on `vmi1227854`.

Median comparison: `282.08 us -> 297.89 us`, a regression.

Score: `0.0` because performance impact was negative.

## Restoration

Production source is restored to HEAD. `git diff -- crates/fsci-stats/src/qmc.rs`
is empty after the RCH gate.
