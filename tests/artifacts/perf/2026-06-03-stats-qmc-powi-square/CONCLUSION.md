# QMC Mixture Square Rewrite Conclusion

Bead: `frankenscipy-qryng`

## Trial

One source lever was trialed in the 2D QMC mixture double-sum hot loop:
replace `delta.powi(2)` with `(delta * delta)` while preserving loop order and
the surrounding arithmetic grouping.

## Behavior Proof

The after RCH `perf_stats qmc-golden` normalized payload matched the before
payload byte-for-byte:

```text
1fb5885cc35367f57b0e818e165a28f87cbb0b9a43fdc7ba4728a6778af44daf
```

Isomorphism obligations preserved during the trial:

- Validation and error order unchanged.
- Sample order unchanged.
- Pair iteration and final accumulation order unchanged.
- Output labels and bit patterns unchanged.
- RNG and tie-breaking surfaces absent.

## Performance Gate

Focused RCH Criterion baseline on `ts2`:

- `qmc_discrepancy/mixture/512x2`: `437.57 us` median

After trial on `vmi1153651`:

- `qmc_discrepancy/mixture/512x2`: `572.31 us` median

The unaffected sibling rows were also slower on the after worker, so this does
not prove a source regression, but it also does not prove a real win. Score is
`0.0` under the campaign keep gate.

## Restore Proof

- `git diff --quiet -- crates/fsci-stats/src/qmc.rs` exited `0` after removing
  the source lever.
- `cargo fmt -p fsci-stats --check` exited `0` after restore.

## Decision

Rejected. The next attack should move away from local 2D QMC arithmetic and
select a different profile-backed subsystem or a stronger exact QMC algorithm
with a bit-identical golden strategy.
