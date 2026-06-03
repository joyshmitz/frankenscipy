# Direct Pinv Sigma Scaling After

- Timestamp: 2026-06-03T17:25:04-04:00
- Bead: `frankenscipy-8l8r1.27`
- Worker: `vmi1264463`
- Command: `RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- 'baseline_pinv/1000x500' --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- Exit: `0`
- Profile-backed target: `baseline_pinv/1000x500`

## Criterion Median

| benchmark | baseline median | after median | result |
| --- | ---: | ---: | --- |
| `baseline_pinv/1000x500` | `316.20 ms` | `1.0569 s` | rejected; `3.34x` slower |

## Behavior Proof

- Before stable SHA-256: `271c9ee685150a31f31ca47867f9b2264eaa254542b1ea49907242bb895bc1cc`
- After stable SHA-256: `271c9ee685150a31f31ca47867f9b2264eaa254542b1ea49907242bb895bc1cc`
- Stable before/after diff: empty.
- Trial contract preserved validation/error order, SVD invocation, singular-value ordering, threshold/rank semantics, certificate fields, output row/column order, RNG absence, tie-breaking absence, and global-state absence.

## Source Restore

- The direct column-scaling source lever was restored after the benchmark failed the keep gate.
- `git diff --quiet -- crates/fsci-linalg/src/lib.rs` exit: `0`
- `cargo fmt -p fsci-linalg --check` after restore exit: `0`

## Gate

Rejected. The trial does not meet Score `>= 2.0` because it produced a measured slowdown despite unchanged behavior proof.
