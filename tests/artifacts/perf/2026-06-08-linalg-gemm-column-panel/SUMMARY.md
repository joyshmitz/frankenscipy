# GEMM column-panel packed RHS keep

Bead: `frankenscipy-v08ky`
Date: 2026-06-08
Target: `fsci-linalg::matmul`, criterion `matmul/1024x1024`

## Profile-backed target

`br ready --json` had no ready perf beads, so this fallback bead targeted the
visible profile hotspot from
`tests/artifacts/perf/2026-06-08-linalg-reprofile-after-eigh-vectors/reprofile_linalg_hotspots_rch.txt`:
`matmul/1024x1024` at `[46.383 ms 48.300 ms 50.660 ms]`.

Alien primitive harvested: affine loop tiling / cache-resident column panels.
The prior GEMM path already packed 8-column RHS panels, but each row tile
reloaded the same A scalars for every 8-column panel. This lever widens the
large-flat-workspace full-tile path from 4x8 to 4x16, consuming two adjacent
packed RHS panels per `k` and reusing each A scalar across both SIMD panel
vectors.

## One lever

Only the large flat-workspace GEMM row kernel changed:

- full tile: `MR=4`, `NR=8`, new `NC=16`
- two packed B panels are loaded for each `k`
- each output element still accumulates `k` in monotonic `0..ka` order
- row split, rectangular dispatch, scalar edge path, RNG behavior, and public
  shape/error behavior are unchanged

## RCH baseline and after

Command:

```text
rch exec -- cargo bench -p fsci-linalg --bench linalg_bench -- matmul/1024x1024
```

Same worker: `fmd`

Before:

```text
matmul/1024x1024        time:   [113.64 ms 122.86 ms 132.09 ms]
```

After:

```text
matmul/1024x1024        time:   [51.753 ms 52.408 ms 53.097 ms]
```

Mean speedup: `122.86 / 52.408 = 2.344x`.
Conservative interval speedup: `113.64 / 53.097 = 2.140x`.

Score: `2.34 impact * 5 confidence / 2 effort = 5.85`, keep.

## Behavior proof

RCH proof command:

```text
rch exec -- cargo test -p fsci-linalg matmul_ -- --nocapture
```

Worker: `vmi1156319`

Passed tests:

```text
test tests::matmul_ikj_is_bit_identical_to_naive_ijk ... ok
test tests::matmul_microkernel_is_bit_identical_to_flat_ikj ... ok
test tests::matmul_flat_compute_rows_row_split_is_bit_identical ... ok
test tests::matmul_flat_workspace_is_bit_identical_to_naive_ijk ... ok
test tests::matmul_microkernel_golden_digest ... ok
```

Isomorphism:

- Ordering: every `c[i][j]` keeps the same `k=0..ka` accumulation order.
- Floating point: each term uses the same separate multiply/add sequence; no
  reordered reductions, no pairwise/tree sum, no fast-math contraction.
- Tie-breaking: no comparisons or tie resolution were introduced.
- RNG: deterministic matrix fixtures and public matmul path use no RNG.
- Output layout: row-major `Vec<Vec<f64>>` assembly is unchanged.

Golden output:

- Rust golden FNV-1a digest unchanged: `0xf9aa16d2dc37468f`
- SHA-256 over the same deterministic 80x80 output f64 little-endian bit stream:
  `2b98ec4be07d4ab1ba51dcd8ac64cf97dde34723ca54f82102c25a97ff534cbd`

## Validation gates

```text
cargo fmt -p fsci-linalg --check
```

Passed locally.

```text
rch exec -- cargo check -p fsci-linalg --all-targets
```

Passed on RCH worker `vmi1264463`.

```text
rch exec -- cargo clippy -p fsci-linalg --all-targets --no-deps -- -D warnings
```

Passed on RCH worker `vmi1149989`.

```text
ubs crates/fsci-linalg/src/lib.rs
```

Critical issues: `0`. Existing broad linalg warnings remain outside this lever.
