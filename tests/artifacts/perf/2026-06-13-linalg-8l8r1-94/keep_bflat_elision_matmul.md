# Kept lever: elide redundant B-flat copy in packed-panel matmul

Bead: `frankenscipy-8l8r1.94`
Date: 2026-06-13

## Target

Profile-backed linalg tail after `frankenscipy-8l8r1.92`: `matmul/1024x1024` was among the top public Criterion tails.

## Lever

The large flat-workspace matmul path already validates `b` as rectangular `Vec<Vec<f64>>` input, then repacked the full matrix into a second row-major `b_flat` buffer before packing only the 8-column micro-panels. This lever removes the redundant full `b_flat` allocation/copy and packs panels directly from the validated input rows. Remainder columns also read the same input rows directly.

No algorithmic operation order changes:

- Output row order is unchanged.
- Each cell still accumulates over `k` in ascending order.
- Packed-panel order remains `jb`, then `k`, then contiguous `j0..j0+8`.
- Tail columns still compute `dj` in ascending order.
- No tie-breaking, RNG, or external BLAS/LAPACK surface is involved.
- The only non-product adjustment is an iterator rewrite in the proof test to satisfy clippy.

## Benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo bench -j 1 -p fsci-linalg --bench linalg_bench -- matmul/1024x1024
```

Same-worker Criterion intervals on `hz1`:

- Baseline: `[115.63 ms 117.64 ms 119.46 ms]`
- After: `[88.421 ms 91.606 ms 94.999 ms]`

Median speedup: `1.2842x`.

Score: `1.2842 impact * 0.90 confidence / 0.50 effort = 2.31`, keep.

## Behavior Proof

Commands:

```text
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo test -j 1 -p fsci-linalg --lib --locked -- matmul --nocapture
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1152480 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo test --release -j 1 -p fsci-linalg --lib --locked -- matmul_medium_flat_workspace_route_golden_digest --ignored --nocapture
```

Results:

- Focused matmul lib tests passed.
- Release golden digest passed with `matmul_medium_flat_route_golden_digest=0x5fd37bf053d54fb0`.
- Golden-output SHA-256: `2c1839a01afde22569567072c466ee858a310c88541a3f146ef00ce7c79f70f4`.

## Validation

- `rustfmt --edition 2024 --check crates/fsci-linalg/src/lib.rs`: passed.
- `git diff --check -- crates/fsci-linalg/src/lib.rs tests/artifacts/perf/2026-06-13-linalg-8l8r1-94`: passed.
- `ubs crates/fsci-linalg/src/lib.rs`: exit 0; no critical findings, broad pre-existing warning inventory recorded.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p fsci-linalg --all-targets --locked`: passed on `hz1`.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1152480 rch exec -- cargo clippy -j 1 -p fsci-linalg --all-targets --no-deps --locked -- -D warnings`: passed on `vmi1152480`.

