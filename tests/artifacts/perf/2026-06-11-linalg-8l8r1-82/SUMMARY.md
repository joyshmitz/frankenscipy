# frankenscipy-8l8r1.82 - GEMM 4x24 Tile Keep

## Target

- Bead: `frankenscipy-8l8r1.82`
- Primitive: widen the existing flat-workspace packed-B full tile from `4x16` to `4x24`.
- Worker for baseline/after: `vmi1227854`.
- Contract: preserve rectangularity gates, fallback paths, row splitting, output row ownership, deterministic traversal, and monotonic `k = 0..ka` accumulation for every output cell.

## Performance

Criterion mean times:

| shape | baseline | after | speedup |
| --- | ---: | ---: | ---: |
| 256x256 | 6.3882 ms | 6.5906 ms | 0.969290x |
| 512x512 | 79.521 ms | 43.950 ms | 1.809352x |
| 768x768 | 136.32 ms | 97.388 ms | 1.399762x |
| 1024x1024 | 284.08 ms | 207.49 ms | 1.369126x |

Affected-size geomean for 512/768/1024: `1.513585x`.

Score: `Impact 1.513585 * Confidence 0.85 / Effort 0.50 = 2.573095`, keep.

## Proof

- `proof_matmul_group_rch.txt`: matmul bit-identity/group proofs passed on `vmi1227854`.
- `proof_medium_route_golden_rch.txt`: release ignored medium-route golden digest passed on `vmi1227854`, digest `0x5fd37bf053d54fb0`.
- `proof_matmul_group_rch_final.txt`: exact-source matmul bit-identity/group proofs passed on `vmi1227854`.
- `proof_medium_route_golden_rch_final.txt`: exact-source release ignored medium-route golden digest passed on `vmi1227854`, digest `0x5fd37bf053d54fb0`.
- `proof_eig_slice_copy_rch.txt`: exact-source focused `eig` proof for the clippy slice-copy cleanup passed on `vmi1227854`.
- `after_4x24_tile_matmul_criterion_rch_final.txt`: exact-source same-worker after benchmark used for the final score.
- `check_fsci_linalg_all_targets_rch.txt`: `cargo check -p fsci-linalg --all-targets --locked` passed on `vmi1227854`.
- `fmt_fsci_linalg_check_final.txt`: `cargo fmt -p fsci-linalg --check` passed.
- `clippy_fsci_linalg_all_targets_no_deps_rch_final.txt`: `cargo clippy -p fsci-linalg --all-targets --no-deps --locked -- -D warnings` passed on `vmi1227854`.
- `ubs_fsci_linalg_lib_final.txt`: UBS exit 0, zero critical issues; broad pre-existing warnings remain in the large linalg file.
- Broad clippy initially exposed pre-existing lints: dependency `fsci-fft` `manual_is_multiple_of`, then non-GEMM linalg `manual_memcpy`. The linalg slice-copy cleanup is included because it is in the changed crate and required for the crate clippy gate; the FFT dependency lint is left out of this perf commit.

## Isomorphism

The GEMM route still packs B panels in the same order and writes the same row-major output slots. The full-tile branch now computes three adjacent 8-column SIMD panels for each 4-row microtile instead of two. Each accumulator still visits `k` in increasing order, so floating-point addition order per output cell is unchanged. There is no RNG, tie-breaking, tolerance, or algorithm-selection surface change.
