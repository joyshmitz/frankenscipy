# fsci-interpolate regular-grid 3D stack fast path

Bead: `frankenscipy-prp1d`

## Target

Profile-backed fallback target:
`regular_grid/linear_eval_many/32x32x16_4096` in
`crates/fsci-interpolate/benches/interpolate_bench.rs`.

Prior profile row:
`tests/artifacts/perf/2026-06-02-interpolate-griddata/reprofile_after_griddata_eval_many_only_broad_rch.txt`
reported `[406.84 us 419.11 us 431.50 us]`.

## Lever

Added a 3D `RegularGridInterpolator::eval_linear` fast path that stores the
three interval indices and fractions in fixed-size stack arrays instead of
allocating two `Vec`s for every query.

The generic N-D path remains unchanged for non-3D inputs.

## Baseline

Command:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-interpolate --bench interpolate_bench --locked -- regular_grid/linear_eval_many/32x32x16_4096 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot
```

Worker: `ts2`

Result:

```text
regular_grid/linear_eval_many/32x32x16_4096 time: [535.38 us 539.86 us 544.30 us]
```

## Behavior Proof

Golden command before:

```text
cargo run -p fsci-interpolate --release --bin perf_interpolate --locked -- regular-grid-linear tests/artifacts/perf/2026-06-04-interpolate-regular-grid-3d-stack/golden_before.txt
```

Golden command after:

```text
cargo run -p fsci-interpolate --release --bin perf_interpolate --locked -- regular-grid-linear tests/artifacts/perf/2026-06-04-interpolate-regular-grid-3d-stack/golden_after.txt
```

SHA-256:

```text
05ba699e8db619b81f33da8def8d654ef68d7e9c4214699e0cb372ac2a25c417  golden_before.txt
05ba699e8db619b81f33da8def8d654ef68d7e9c4214699e0cb372ac2a25c417  golden_after.txt
```

`cmp -s golden_before.txt golden_after.txt` passed.

Isomorphism:

- Ordering: unchanged query input order and corner loop order `0..8`.
- Tie-breaking: unchanged `find_interval` and bounds/fill handling.
- Floating point: unchanged denominator/fraction arithmetic, dimension order,
  flat-index arithmetic, weight multiplication order, and accumulation order.
- RNG: no RNG path is used by this benchmark or golden harness.

## Rebench

Command:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-interpolate --bench interpolate_bench --locked -- regular_grid/linear_eval_many/32x32x16_4096 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot
```

Worker: `ts2`

Result:

```text
regular_grid/linear_eval_many/32x32x16_4096 time: [270.79 us 271.98 us 273.08 us]
```

Median speedup: `539.86 / 271.98 = 1.985x`.

Score: `10.0 = impact 4 * confidence 5 / effort 2`; keep.

## Validation

Passed:

```text
cargo fmt -p fsci-interpolate --check
ubs crates/fsci-interpolate/src/lib.rs crates/fsci-interpolate/src/bin/perf_interpolate.rs
RCH_FORCE_REMOTE=1 rch exec -- cargo check -p fsci-interpolate --all-targets --locked
RCH_FORCE_REMOTE=1 rch exec -- cargo clippy -p fsci-interpolate --all-targets --locked -- -D warnings
RCH_FORCE_REMOTE=1 rch exec -- cargo test -p fsci-interpolate --locked regular_grid
```

The focused test run covered 17 regular-grid unit tests plus
`mr_regular_grid_interpolator_passes_through_grid`.

## Post-change Reprofile

Command:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-interpolate --bench interpolate_bench --locked -- regular_grid --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot
```

Worker: `ts2`

Results:

```text
regular_grid/linear_eval_many/32x32x16_4096  time: [271.45 us 271.75 us 272.30 us]
regular_grid/nearest_eval_many/32x32x16_4096 time: [353.73 us 354.65 us 355.53 us]
```

Next profiler-visible target in this group: `nearest_eval_many/32x32x16_4096`.
