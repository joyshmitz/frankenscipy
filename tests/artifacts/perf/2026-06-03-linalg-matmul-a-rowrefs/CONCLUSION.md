# A row-reference matmul trial conclusion

Bead: `frankenscipy-8l8r1.11`
Target: `fsci-linalg::matmul`, existing 4x4 register micro-kernel.

## Lever

Trial lever: hoist the four `A` row references once per full 4x4 tile, then load `a_rowN[k]` inside the existing `k` loop. This kept direct `B` row indexing and did not add allocation, packing, or loop-order changes.

Source status: rejected and restored. `source_restored_diff.txt` is empty for `crates/fsci-linalg/src/lib.rs` and `crates/fsci-linalg/benches/linalg_bench.rs`.

## Baseline and profile

Fresh RCH Criterion baseline for `matmul` on `vmi1153651`:

| Size | Baseline median |
| --- | ---: |
| 256x256 | 12.326 ms |
| 512x512 | 148.22 ms |
| 768x768 | 1.0100 s |
| 1024x1024 | 2.0473 s |

The after run on a different worker was much faster, so it was treated as cross-worker noise and not accepted by itself.

## Exact paired benchmark

A temporary bench-only paired comparator measured the exact prior `a[i][k]` tile and the row-reference candidate in one RCH run on `vmi1227854`.

| Size | Prior median | Row-ref median | Speedup |
| --- | ---: | ---: | ---: |
| 256x256 | 3.2163 ms | 3.9890 ms | 0.81x |
| 512x512 | 23.783 ms | 35.580 ms | 0.67x |
| 768x768 | 80.176 ms | 122.86 ms | 0.65x |
| 1024x1024 | 389.95 ms | 594.60 ms | 0.66x |

Result: rejected. Every paired size regressed.

## Behavior proof

Golden-output proof before and after the trial kept normalized sha256:

`0def10fbd95d1bf20c417af563de181eeab314cae762cc82fd67c1ebac6f406c`

The trial preserved the intended isomorphism while applied: API/error behavior, validation order, output order, per-cell monotonic `k` accumulation, separate multiply/add operations, direct `B` row access, scalar ragged path, RNG absence, tie-breaking absence, and global-state absence were unchanged. After rejection, production source was restored.

## Validation

- RCH `cargo test -p fsci-linalg --release matmul_microkernel --locked -- --nocapture` passed before the trial.
- RCH `cargo test -p fsci-linalg --release matmul_microkernel --locked -- --nocapture` passed after the trial.
- RCH `cargo test -p fsci-linalg --release matmul_microkernel --locked -- --nocapture` passed after restore.
- `cargo fmt -p fsci-linalg --check` passed after edit and after restore.
- `source_restored_diff.txt` is empty.

## Score

Score: `0.0`. Performance impact was negative under the exact paired benchmark, so the lever did not meet the required `>=2.0` keep threshold.

## Next profile target

Re-profile before selecting another lever. This result suggests LLVM already optimizes the direct `a[i][k]` indexing better than the pre-borrowed row references for this loop shape.
