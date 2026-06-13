# fsci-linalg matmul SIMD store lever

Bead: `frankenscipy-l566o`

Target: profiler-evident large GEMM path used by the linalg no-gaps/LU campaign.
The edited function is `matmul_flat_compute_rows`, specifically the full SIMD
tile store epilogue for `NR=8` and `NC=24`.

## Lever

Replace scalar lane extraction stores:

- `Simd::to_array()` into temporary arrays
- individual scalar assignments for each lane

with direct vector lane stores:

- `Simd::copy_to_slice(&mut out[..])`

No arithmetic, loop bounds, blocking, route selection, row splitting, packing,
threading, or matrix generation changed.

## Benchmark

Criterion via RCH, crate-scoped command:

`cargo bench -j 1 -p fsci-linalg --bench linalg_bench -- matmul/1024x1024 --sample-size 20`

Same worker: `vmi1227854`.

| run | mean | interval |
| --- | ---: | --- |
| baseline | 66.828 ms | [64.768 ms, 69.058 ms] |
| after | 60.069 ms | [57.314 ms, 64.710 ms] |

Speedup: `1.1125x` by mean.

Score: `2.85 = Impact 3.0 * Confidence 0.95 / Effort 1.0`.

## Isomorphism proof

- Ordering: unchanged. The `k in 0..ka` accumulation order and row/column tile
  traversal are byte-for-byte the same as before.
- Floating point: unchanged. The patch performs no floating-point arithmetic and
  does not reassociate, fuse, skip, or reorder operations. It stores the same
  `Simd<f64, 8>` lanes directly instead of materializing the same lanes through
  `to_array()`.
- Tie-breaking: none in this code path.
- RNG: none. Bench/test matrices are deterministic arithmetic fixtures and no
  random source is touched.
- Threading/row split: unchanged. Existing row-split bit-identity proof passed.

## Behavior and gates

- `rustfmt --edition 2024 --check crates/fsci-linalg/src/lib.rs`: pass.
- `ubs crates/fsci-linalg/src/lib.rs`: exit 0. UBS reports existing warning
  inventories in this large file, with no critical findings; its cargo fmt,
  clippy, check, and test-build subchecks were clean.
- `cargo test -j 1 -p fsci-linalg --release --lib matmul -- --nocapture` via
  RCH: pass. The run includes:
  - `matmul_flat_workspace_is_bit_identical_to_naive_ijk`
  - `matmul_microkernel_is_bit_identical_to_flat_ikj`
  - `matmul_ikj_is_bit_identical_to_naive_ijk`
  - `matmul_microkernel_golden_digest`
  - `matmul_flat_compute_rows_row_split_is_bit_identical`
- `cargo test -j 1 -p fsci-linalg --release --lib matmul_medium_flat_workspace_route_golden_digest -- --ignored --nocapture`
  via RCH: pass, golden digest `0x5fd37bf053d54fb0`.
- `cargo check -j 1 -p fsci-linalg --all-targets` via RCH: pass.
- `cargo clippy -j 1 -p fsci-linalg --all-targets --no-deps -- -D warnings`
  via RCH: pass.

## Artifacts

- `baseline_matmul_1024_vmi1152480_rch.txt` (RCH selected `vmi1227854`)
- `after_matmul_1024_vector_store_vmi1227854_rch.txt`
- `proof_matmul_route_and_golden_rch.txt`
- `proof_matmul_medium_flat_route_golden_rch.txt`
- `rustfmt_linalg_lib_check.txt`
- `ubs_linalg_lib_vector_store.txt`
- `check_fsci_linalg_all_targets_rch.txt`
- `clippy_fsci_linalg_all_targets_no_deps_rch.txt`
- `sha256sums.txt`
