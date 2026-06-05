# QMC Row-Major Formula-Cost Reduction Rejection

Bead: `frankenscipy-ezwvj`

## Target

The fallback profile-backed target was the 2D QMC discrepancy group after the
Halton bead was already closed and `br ready --json` was empty.

Fresh focused RCH baseline:

- Worker: `ts1`
- Command: `RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-stats --bench stats_bench --locked -- qmc_discrepancy --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot`
- `centered/512x2`: `244.01 us` median
- `mixture/512x2`: `293.81 us` median
- `l2_star/512x2`: `227.18 us` median
- `wraparound/512x2`: `216.19 us` median

## Lever Tried

The trial removed identity `1.0 * term` product scaffolding inside the 2D
discrepancy helpers while preserving loop order, coordinate order, term
expressions, and the final `single +=` / `double +=` accumulation points.

This was not a repeat of the prior rejected QMC levers:

- symmetric upper-triangle pair cache / replay
- broad point SoA
- coordinate-only L2-star cache
- direct `powi(2)` square rewrite

## Behavior Proof

The QMC golden payload before and after was byte-identical.

SHA-256:

```text
1fb5885cc35367f57b0e818e165a28f87cbb0b9a43fdc7ba4728a6778af44daf
```

Preserved surfaces:

- validation and error order
- sample row order
- pair loop order
- coordinate order
- term expressions
- final accumulation order
- RNG absence
- tie-breaking absence

## Performance Gate

The after and confirmation runs both landed on `ts2`, so the fresh `ts1`
baseline was not same-worker comparable. Against the prior current-code `ts2`
QMC baseline from `frankenscipy-pgm85`, the candidate was flat to slower:

- prior `ts2` `mixture/512x2`: `437.57 us`
- after `ts2` `mixture/512x2`: `449.28 us`
- confirm `ts2` `mixture/512x2`: `446.20 us`

Sibling rows were also flat to slower on confirmation:

- `centered/512x2`: `320.01 us`
- `l2_star/512x2`: `337.49 us`
- `wraparound/512x2`: `332.59 us`

The trial therefore did not prove a real win and fails the Score >= 2.0 keep
gate.

## Restore Proof

- Production `crates/fsci-stats/src/qmc.rs` was restored.
- `git diff --quiet -- crates/fsci-stats/src/qmc.rs` exited `0`.
- `cargo fmt -p fsci-stats --check` exited `0`.

## Verdict

Rejected. Do not repeat identity-product scaffolding removal as a QMC
discrepancy lever. The next stats pass should pick a different profile-backed
hotspot or a fundamentally different QMC primitive.
