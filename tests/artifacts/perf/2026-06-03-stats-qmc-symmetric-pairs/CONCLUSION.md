# QMC Symmetric Pair Cache Conclusion

Bead: `frankenscipy-pgm85`

## Trial

One source lever was trialed: a bounded upper-triangle pair-product cache for
the 2D L2-star discrepancy double sum, followed by row-major replay of the
final accumulation.

## Behavior Proof

The after RCH `perf_stats qmc-golden` normalized payload matched the before
payload byte-for-byte:

```text
1fb5885cc35367f57b0e818e165a28f87cbb0b9a43fdc7ba4728a6778af44daf
```

Isomorphism obligations preserved during the trial:

- Validation and error order unchanged.
- Sample order unchanged.
- Final double-sum accumulation replayed original row-major `(i, j)` order.
- Output labels and bit patterns unchanged.
- RNG and tie-breaking surfaces absent.

## Performance Gate

Focused RCH Criterion baseline:

- `qmc_discrepancy/l2_star/512x2`: `329.44 us` median

After trial:

- `qmc_discrepancy/l2_star/512x2`: `1.2924 ms` median

The trial was a `3.92x` slowdown, so Score is `0.0`.

## Restore Proof

- `git diff --quiet -- crates/fsci-stats/src/qmc.rs` exited `0` after removing
  the source lever.
- `cargo fmt -p fsci-stats --check` exited `0` after restore.

## Decision

Rejected. The next QMC attempt must attack a different primitive, such as
row-major formula-cost reduction that preserves per-pair accumulation order, or
a mathematically stronger exact algorithm only if its golden output is
bit-identical under the project proof gate.
