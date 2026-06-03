# QMC L2-Star Coordinate-Only Cache Negative Result

Bead: `frankenscipy-ilfbq`

## Verdict

Rejected. The l2-star coordinate-only cache regressed the focused RCH benchmark,
so production source was restored and no code change is kept.

## Profile-Backed Target

Source profile:

`tests/artifacts/perf/2026-06-03-stats-qmc-point-soa/CONCLUSION.md`

Relevant row:

- `qmc_discrepancy/l2_star/512x2`: `219.44 us` median in the prior focused QMC baseline.

Fresh focused baseline for this bead:

- `qmc_discrepancy/l2_star/512x2`: `[224.20 us, 228.55 us, 233.04 us]` on `vmi1149989`.

## One Lever Tested

Added a private coordinate-only 2D helper and routed only
`l2_star_discrepancy_2d` through it, avoiding the full centered/abs point cache.

The previously rejected broad SoA layout and direct `powi(2)` replacement were
not repeated.

## Behavior Surface

The intended preserved surfaces were:

- Validation order and error surfaces.
- Row-major sample order.
- Pair loop order: `i` outer, `j` inner.
- Coordinate order: 0 then 1.
- Formula term order and final `sqrt`.
- RNG absence, tie-breaking absence, and global-state absence.

Golden before and after:

`1fb5885cc35367f57b0e818e165a28f87cbb0b9a43fdc7ba4728a6778af44daf`

The after-golden RCH job produced the same QMC block byte-for-byte. The source
was still restored because the focused RCH benchmark regressed.

## Benchmark Gate

Focused RCH after:

- `qmc_discrepancy/l2_star/512x2`: `[470.18 us, 482.77 us, 496.97 us]` on `vmi1153651`.

Median comparison: `228.55 us -> 482.77 us`, a regression.

Score: `0.0` because performance impact was negative.

## Restoration

Production `crates/fsci-stats/src/qmc.rs` is restored to HEAD.

- `source_restored_diff.txt`: empty.
- `cargo fmt -p fsci-stats --check`: pass.
