# frankenscipy-grljp column-major Householder rank-2 update

Bead: `frankenscipy-grljp`
Agent: `RubyWaterfall`
Date: 2026-06-15
Base: `a9dbf56e` plus the one source lever in `crates/fsci-linalg/src/lib.rs`

## Profile-backed target

The target is the native symmetric-eigh Householder reduction stage. Prior stage evidence at n=1200 showed:

- Householder reduction: `406.741 ms`
- Tridiagonal eigen: `75.736 ms`
- Eigenvector back-transform: `103.695 ms`
- Sort/copy: `5.408 ms`

The previous per-step p-vector threading subprobe was rejected. This keep avoids thread spawning and changes only the memory traversal of the symmetric rank-2 update.

## Lever

`apply_symmetric_householder_trailing_rank2` now writes the trailing update by column:

- `p = tau * A_active * v` is unchanged.
- `w = p - 0.5 * tau * (v dot p) * v` is unchanged.
- Each lower-triangle entry still applies `A_ij -= v_i*w_j + w_i*v_j`.
- The symmetric mirror is assigned the same computed value.
- Diagonal entries avoid the redundant mirror write.

This preserves the scalar formula per entry and improves locality for the stored lower-triangle write.

## Behavior proof

- RCH `symmetric_rank2_column_update_matches_rowwise_bits` passed against an in-test rowwise reference and compared `p`, `w`, and every matrix entry by `f64::to_bits()`.
- RCH `symmetric_eigh_native_matches_nalgebra_and_timing` passed.
- RCH `eigh_index_sort_matches_materialized_pair_sort_bits` passed and kept public digest `0x287a5d3679a8bc6a`.
- Ordering, tie-breaking, RNG inputs, fallback policy, and public route thresholds were unchanged.

## Benchmark

Same usable RCH worker: `ovh-a`.

Baseline routed native:

- n=400 `56.673791 ms`
- n=800 `253.243413 ms`
- n=1200 `1038.249562 ms`

Candidate native timing:

- n=400 `40.2 ms`
- n=800 `217.8 ms`
- n=1200 `666.2 ms`

Score: `Impact 3.0 * Confidence 4.0 / Effort 2.0 = 6.0`.

## Gates

- RCH `cargo check -j 1 -p fsci-linalg --lib --locked`: passed.
- `cargo fmt -p fsci-linalg -- --check`: passed after formatting the touched file.
- UBS `crates/fsci-linalg/src/lib.rs`: `Critical issues: 0`.
- RCH `cargo clippy -j 1 -p fsci-linalg --lib --no-deps --locked -- -D warnings`: blocked only by pre-existing `needless_range_loop` findings at `lib.rs:3709`, `3720`, and `4170`; the candidate-specific lint was fixed and did not recur.

Verdict: keep.
