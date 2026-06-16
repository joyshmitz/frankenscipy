# frankenscipy-psn7x.8 compact backtransform replay

Agent: RubyWaterfall
Worker: `vmi1227854`
Scope: `fsci-linalg` symmetric `eigh` native backtransform
Traceability: supplemental keep for a distinct compact-WY chunk replay after
upstream `frankenscipy-psn7x.5` had already closed rejected for a column-pair
backtransform replay and upstream `frankenscipy-psn7x.7` was claimed for the
next two-stage reduction path. The artifact directory name retains the original
target path used before the rebase.

## Lever

Prebuild safe-Rust compact-WY panels for large symmetric-eigh Householder
backtransforms and replay each panel per existing column chunk. Scalar replay
stays as the fallback for small matrices or failed panel construction.

## Baseline

Command:

```text
env RCH_WORKER=vmi1227854 rch exec -- cargo test -j 1 -p fsci-linalg --lib symmetric_eigh_native_stage_breakdown_probe --release --locked -- --ignored --nocapture
```

Stage timings before the lever:

| shape | reduction_ms | tridiagonal_eigen_ms | backtransform_ms | sort_ms | values_digest |
| --- | ---: | ---: | ---: | ---: | --- |
| 400x400 | 29.986636 | 11.492557 | 44.046860 | 0.555172 | `0x0dbbde75b75c8612` |
| 800x800 | 211.072922 | 52.806563 | 119.698658 | 4.512491 | `0x4461962827bdb038` |
| 1200x1200 | 449.898201 | 90.041898 | 432.521169 | 7.082561 | `0x2fc45e1f18ceb0ab` |

The stage run showed worker-load drift, including the 400x400 control path where
the compact lever is disabled. The decisive keep/reject evidence is therefore
the same-binary scalar-vs-compact chunk probe below.

## Same-Binary Kernel A/B

Command:

```text
env RCH_WORKER=vmi1227854 rch exec -- cargo test -j 1 -p fsci-linalg --lib compact_wy_backtransform_chunk_perf_probe --release --locked -- --ignored --nocapture
```

| shape | chunk_cols | scalar_chunk_ms | compact_chunk_ms | speedup | max_abs_diff | values_digest |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 400x400 | 50 | 3.279484 | 3.067676 | 1.069045 | 1.47104550762833242e-15 | `0x0dbbde75b75c8612` |
| 800x800 | 100 | 24.939755 | 23.156474 | 1.077010 | 1.81799020282369383e-15 | `0xad8a7e5fa1980bfb` |
| 1200x1200 | 150 | 84.808904 | 80.609583 | 1.052095 | 1.79023462720806492e-15 | `0x181b3486089d0e4a` |

Score: Impact 2.5 x Confidence 0.9 / Effort 1.0 = 2.25. Kept.

## After Stage Probe

Command:

```text
env RCH_WORKER=vmi1227854 rch exec -- cargo test -j 1 -p fsci-linalg --lib symmetric_eigh_native_stage_breakdown_probe --release --locked -- --ignored --nocapture
```

| shape | reduction_ms | tridiagonal_eigen_ms | backtransform_ms | sort_ms | values_digest |
| --- | ---: | ---: | ---: | ---: | --- |
| 400x400 | 18.760057 | 10.176986 | 7.899245 | 0.618747 | `0x0dbbde75b75c8612` |
| 800x800 | 107.671528 | 38.820221 | 43.708705 | 1.896391 | `0x4461962827bdb038` |
| 1200x1200 | 336.224240 | 87.673585 | 130.295112 | 5.474692 | `0x2fc45e1f18ceb0ab` |

Because the 400x400 control path also moved, these stage timings are routing
evidence only. The same-binary A/B table is the keep evidence.

## Isomorphism Proof

Ordering and tie-breaking: unchanged. Eigenvalue ordering still uses the
existing `total_cmp` sort and column reorder after backtransform.

Floating point: the compact-WY path changes arithmetic association inside the
backtransform, so eigenvector bits are not expected to match the scalar replay.
The added proof fixture bounds deterministic scalar-vs-compact drift:

```text
compact_wy_backtransform_contract n=48 panel_width=4 max_abs_diff=1.11022302462515654e-15 scalar_digest=0x1db8d23e5f1bfdf5 compact_digest=0x49be718d438225e1
compact_wy_backtransform_contract n=96 panel_width=8 max_abs_diff=1.49880108324396133e-15 scalar_digest=0x9f470b07af18922e compact_digest=0x4df0ef983ffb6a82
```

Public golden output: eigenvalue digests and tolerance checks stayed unchanged
for 400/800/1200. Golden output file:
`tests/artifacts/perf/2026-06-16-rubywaterfall-linalg-psn7x5-compact-backtransform/golden-public-values.txt`

Golden output sha256:

```text
45b6e4a1f1ab010d093779bb32ed085aab49030617bbb9f125e6e05ff8b27627
```

RNG: no production RNG. Probes use the same deterministic LCG seeds as the
existing stage/public route probes.

Safety: safe Rust only; no `unsafe` added.

## Public Route Proof

Command:

```text
env RCH_WORKER=vmi1227854 rch exec -- cargo test -j 1 -p fsci-linalg --lib public_eigh_native_route_perf_probe --release --locked -- --ignored --nocapture
```

| shape | routed_eigh_ms | nalgebra_eigh_ms | speedup | max_abs_diff | routed_values_digest |
| --- | ---: | ---: | ---: | ---: | --- |
| 400x400 | 45.111726 | 48.609049 | 1.077526 | 0.00000000000000000e0 | `0x4b8334c92ce624eb` |
| 800x800 | 198.012765 | 338.984565 | 1.711933 | 1.50635059981141239e-12 | `0xad8a7e5fa1980bfb` |
| 1200x1200 | 666.105475 | 1156.406022 | 1.736070 | 2.05346850634668954e-12 | `0x181b3486089d0e4a` |

## Gates

Passed:

- `ubs crates/fsci-linalg/src/lib.rs` (exit 0; warnings are pre-existing inventory)
- `cargo fmt -p fsci-linalg -- --check`
- `env RCH_WORKER=vmi1227854 rch exec -- cargo check -j 1 -p fsci-linalg --lib --locked`
- `env RCH_WORKER=vmi1227854 rch exec -- cargo clippy -j 1 -p fsci-linalg --lib --no-deps --locked -- -D warnings -A clippy::manual_is_multiple_of -A clippy::needless_range_loop -A clippy::mut_range_bound`

Known unrelated blockers:

- Strict `env RCH_WORKER=vmi1227854 rch exec -- cargo clippy -j 1 -p fsci-linalg --all-targets --no-deps --locked -- -D warnings`
  is blocked by pre-existing lints in `crates/fsci-linalg/src/lib.rs` around
  `12587-12971` (`rsf2csf`/`matrix_balance`). This is tracked as
  `frankenscipy-8ykh7`.
- Dependency checking still reports the pre-existing `fsci-fft` unused variable
  warning at `crates/fsci-fft/src/helpers.rs:58`.
