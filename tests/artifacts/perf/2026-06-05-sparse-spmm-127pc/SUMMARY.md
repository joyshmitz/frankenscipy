# frankenscipy-127pc SpMM fused chunk pass

## Target

- Bead: `frankenscipy-127pc`
- Crate: `fsci-sparse`
- Benchmark: `sparse_spmm/2000x2000_d1/2000`
- Lever: fuse each work-balanced SpMM chunk's symbolic count and numeric fill inside the same worker, keeping the existing row-local numeric kernel and concatenation order.

## Benchmark

- Baseline: RCH `ts2`, median `13.405 ms`, interval `[13.333 ms, 13.481 ms]`
- After: RCH `ts2`, median `11.632 ms`, interval `[11.417 ms, 11.838 ms]`
- Delta: `-1.773 ms`, `13.2%` faster by median.
- Score: Impact `3` x Confidence `3` / Effort `1` = `9.0`, keep.

## Isomorphism Proof

- Row order: unchanged; ranges are still produced by `spmm_work_balanced_ranges` and concatenated in range order.
- A row traversal: unchanged; each row still calls `spmm_row_counts_chunk` and `spmm_row_chunk` over the same row bounds.
- B encounter order: unchanged; numeric fill still uses `spmm_row_chunk`.
- Floating-point order: unchanged; the same numeric kernel performs the same multiply/add sequence per row.
- Zero elision: unchanged; `spmm_row_counts_chunk` remains the symbolic count and `debug_assert_eq!` verifies numeric counts match.
- Metadata: unchanged; sorted flag is still the conjunction of chunk sorted flags.
- RNG: absent from the production path.

## Golden Proof

- Previous strict payload: `tests/artifacts/perf/2026-06-05-sparse-spmm-work-balanced-rows/golden_after_payload.strict.txt`
- Current strict payload: `golden_after_payload.strict.txt`
- `cmp -s` exit: `0`
- SHA256: `0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2`

## Validation

- `RCH_FORCE_REMOTE=1 rch exec -- cargo check -p fsci-sparse --all-targets --locked`: pass
- `RCH_FORCE_REMOTE=1 rch exec -- cargo clippy -p fsci-sparse --all-targets --locked -- -D warnings`: pass
- `RCH_FORCE_REMOTE=1 rch exec -- cargo test -p fsci-sparse --locked -- --nocapture`: pass, `309 passed`, `1 ignored`, plus `56` metamorphic tests
- `cargo fmt -p fsci-sparse --check`: pass
- `ubs crates/fsci-sparse/src/linalg.rs`: exit `0`, no critical issues

## Next

Reprofile after landing. If SpMM remains first, the next deeper primitive should be a true CSC/column-panel or semiring-symbolic SpGEMM lane, not another scheduling or finalization tweak.
