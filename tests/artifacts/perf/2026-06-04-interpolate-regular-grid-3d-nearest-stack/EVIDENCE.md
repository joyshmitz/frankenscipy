# fsci-interpolate regular-grid 3D nearest fast path

Bead: `frankenscipy-a6j30`

## Profile-backed target

After `frankenscipy-prp1d`, the post-change RCH regular-grid sweep on `ts2` made nearest the slower regular-grid path:

- `regular_grid/linear_eval_many/32x32x16_4096`: `[271.45 us 271.75 us 272.30 us]`
- `regular_grid/nearest_eval_many/32x32x16_4096`: `[353.73 us 354.65 us 355.53 us]`

The hot path still paid the generic per-dimension iterator/indexing path for a fixed 3D nearest workload.

## Lever

Add a fixed-rank 3D nearest `eval_many` fast path with stack/register state. The generic path is retained for other dimensionalities and methods.

## RCH baseline

Command:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-interpolate --bench interpolate_bench --locked -- regular_grid/nearest_eval_many/32x32x16_4096 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot
```

Worker: `ts2`

Result:

```text
regular_grid/nearest_eval_many/32x32x16_4096
time: [349.90 us 351.21 us 352.23 us]
```

## Golden proof

Golden command:

```text
cargo run -p fsci-interpolate --release --bin perf_interpolate --locked -- regular-grid-nearest tests/artifacts/perf/2026-06-04-interpolate-regular-grid-3d-nearest-stack/golden_after.txt
```

Before and after SHA-256:

```text
359e534a5e8d13b951881603c7df58ff2a249e2186cdb823e24bfe9157713ecf  golden_before.txt
359e534a5e8d13b951881603c7df58ff2a249e2186cdb823e24bfe9157713ecf  golden_after.txt
```

`cmp -s golden_before.txt golden_after.txt` passed.

## Isomorphism notes

- Query order and output order are unchanged.
- Dimension validation remains before evaluation with the same `expected 3D, got {n}D` error text.
- NaN handling remains before bounds checks and returns `f64::NAN`.
- Bounds checks preserve dimension order, fill-value behavior, and out-of-bounds error formatting.
- Interval selection still calls `find_interval`.
- Nearest tie-breaking is unchanged: advance only when `(x - axis[i]).abs() > (axis[i + 1] - x).abs()`.
- Flat-index accumulation preserves dimension order and stride arithmetic.
- No RNG or floating-point arithmetic changes beyond the same comparisons and index selection.

## RCH rebench

First after-run was cross-worker only:

- Worker: `vmi1227854`
- Result: `[151.42 us 158.13 us 166.65 us]`

Same-worker confirmation:

Command:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-interpolate --bench interpolate_bench --locked -- regular_grid/nearest_eval_many/32x32x16_4096 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot
```

Worker: `ts2`

Result:

```text
regular_grid/nearest_eval_many/32x32x16_4096
time: [283.83 us 295.33 us 303.45 us]
```

Same-worker median speedup: `351.21 us / 295.33 us = 1.189x`.

Score: `4.0` (`impact=2`, `confidence=4`, `effort=2`), so the lever is retained.

## Validation

```text
cargo fmt -p fsci-interpolate --check
git diff --check
ubs crates/fsci-interpolate/src/lib.rs crates/fsci-interpolate/src/bin/perf_interpolate.rs
RCH_FORCE_REMOTE=1 rch exec -- cargo check -p fsci-interpolate --all-targets --locked
RCH_FORCE_REMOTE=1 rch exec -- cargo clippy -p fsci-interpolate --all-targets --locked -- -D warnings
RCH_FORCE_REMOTE=1 rch exec -- cargo test -p fsci-interpolate --locked regular_grid
```

All commands passed. `ubs` exited 0 with the existing broad warning inventory only.

## Post-change reprofile

Command:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-interpolate --bench interpolate_bench --locked -- regular_grid --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot
```

Worker: `ts2`

Results:

```text
regular_grid/linear_eval_many/32x32x16_4096
time: [272.18 us 275.69 us 280.38 us]

regular_grid/nearest_eval_many/32x32x16_4096
time: [290.98 us 299.56 us 302.79 us]
```
