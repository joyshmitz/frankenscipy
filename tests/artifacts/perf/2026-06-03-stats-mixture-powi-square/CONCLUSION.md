# Mixture 2D Powi-Square Negative Result

Bead: `frankenscipy-8l8r1.5`

## Verdict

Abandoned. The direct square lever was byte-identical but regressed the focused
RCH benchmark, so production source was restored and no code change was kept.

## Profile-Backed Target

Source profile:

`tests/artifacts/perf/2026-06-03-stats-sobol-directions/reprofile_stats_after_sobol_rch.txt`

The shifted stats profile included:

- `qmc_discrepancy/mixture/512x2`: `627.41us` median.

Focused baseline:

- `qmc_discrepancy/mixture/512x2`: `[255.36 us, 267.89 us, 274.25 us]` on `vmi1227854`.

## One Lever Tested

Replaced `delta.powi(2)` with `delta * delta` in the 2D mixture discrepancy
pair loop only.

## Behavior Proof

The QMC golden before and after the candidate was byte-identical.

SHA256:

`1fb5885cc35367f57b0e818e165a28f87cbb0b9a43fdc7ba4728a6778af44daf`

Preserved surfaces:

- Validation order.
- Row-major sample order.
- Pair loop order: `point_i` outer, `point_j` inner.
- Coordinate order: 0 then 1.
- Formula term order.
- RNG, tie-breaking, and global state.

## Benchmark Gate

Focused RCH after:

- `qmc_discrepancy/mixture/512x2`: `[675.55 us, 733.13 us, 795.27 us]` on `vmi1153651`.

Result by medians: `0.37x`; regression.

Score: `0.0` because impact was negative.

## Restoration

Production `crates/fsci-stats/src/qmc.rs` was restored to the pre-lever
implementation. `git diff -- crates/fsci-stats/src/qmc.rs` is empty.
