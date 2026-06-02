# fsci-sparse CSR/CSC Conversion Optimization Report

Bead: `frankenscipy-572zu`

## Target

Profile-backed hotspot:

- Broad pre-profile on RCH worker `vmi1227854`: `sparse_format_conversion/10000x10000_d0_csr_to_csc/10000` mean `7.9314 ms`.
- Broad pre-profile on RCH worker `vmi1227854`: `sparse_format_conversion/10000x10000_d0_csc_to_csr/10000` mean `9.1720 ms`.
- Handoff: `tests/artifacts/perf/2026-06-02-sparse-profile/hotspot_table.md`.

## One Lever

Use direct compressed transpose for canonical CSR/CSC matrices. Non-canonical or structurally questionable inputs keep the existing COO canonicalization path.

Opportunity score: `Impact 5 * Confidence 5 / Effort 3 = 8.33`.

## Benchmark Result

| Row | Before | After | Delta |
|-----|--------|-------|-------|
| `csr_to_csc/10000` | `5.9070 ms` focused baseline on `vmi1293453` | `597.97 us` post-profile on `vmi1293453` | `9.88x faster` |
| `csc_to_csr/10000` | `5.7233 ms` focused baseline on `vmi1293453` | `561.20 us` post-profile on `vmi1293453` | `10.20x faster` |
| `csr_to_csc/10000` | `7.9314 ms` broad pre-profile on `vmi1227854` | `724.55 us` focused after on `vmi1227854` | `10.95x faster` |
| `csc_to_csr/10000` | `9.1720 ms` broad pre-profile on `vmi1227854` | `886.01 us` focused after on `vmi1227854` | `10.35x faster` |

Artifacts:

- Baseline: `tests/artifacts/perf/2026-06-02-sparse-format-conversion/baseline_conversion_10000_rch.txt`.
- After: `tests/artifacts/perf/2026-06-02-sparse-format-conversion/after_conversion_10000_rch.txt`.
- Post-profile: `tests/artifacts/perf/2026-06-02-sparse-format-conversion/post_profile_sparse_bench_rch.txt`.

## Isomorphism Proof

- Ordering preserved: yes. Canonical CSR rows are scanned in ascending row order and appended into CSC column cursors, so each CSC column receives ascending row indices. Canonical CSC columns are scanned in ascending column order and appended into CSR row cursors, so each CSR row receives ascending column indices.
- Tie-breaking unchanged: yes. The direct path is gated to sorted and deduplicated compressed inputs. Duplicate, unsorted, malformed, or uncertain inputs fall back to the existing `to_coo()?.to_csr()/to_csc()` path, preserving duplicate summation behavior.
- Floating-point: identical for the direct path. Values are copied by assignment with no arithmetic, preserving `f64::to_bits()` including `-0.0` and payload bits. The fallback path remains unchanged for duplicate summation.
- RNG seeds: unchanged. Bench inputs still use the existing `SEED` in `sparse_bench.rs`.
- Golden outputs: before and after snapshot files are byte-identical. SHA256: `f01e261f50d39eab13c364c8af2dee85d335ad78729e18d2014dfa17450d2efe`.

Golden artifacts:

- `tests/artifacts/perf/2026-06-02-sparse-format-conversion/golden_before.txt`.
- `tests/artifacts/perf/2026-06-02-sparse-format-conversion/golden_after.txt`.
- `tests/artifacts/perf/2026-06-02-sparse-format-conversion/golden_before.sha256`.
- `tests/artifacts/perf/2026-06-02-sparse-format-conversion/golden_after.sha256`.

## Validation

- RCH `cargo test -p fsci-sparse --lib conversion_golden_snapshot --locked -- --nocapture`: passed before and after.
- RCH `cargo test -p fsci-sparse --lib conversion --locked -- --nocapture`: passed `10/10`.
- `rustfmt --check crates/fsci-sparse/src/ops.rs`: passed.
- `ubs crates/fsci-sparse/src/ops.rs`: exit `0`.
- RCH `cargo clippy -p fsci-sparse --all-targets --locked -- -D warnings`: exit `101`, failing only on pre-existing lowercase SciPy-compatible type aliases in `crates/fsci-sparse/src/lib.rs`.
- RCH supplementary `cargo clippy -p fsci-sparse --all-targets --locked -- -A non_camel_case_types -D warnings`: passed.

## Shifted Bottleneck

Post-change broad sparse profile on RCH worker `vmi1293453`:

| Rank | Row | Mean |
|------|-----|------|
| 1 | `sparse_arithmetic/10000x10000_d0_add/10000` | `1.6954 ms` |
| 2 | `sparse_csr_construction/10000x10000_d0/10000` | `1.6720 ms` |
| 3 | `sparse_diags/tridiag/10000` | `1.2442 ms` |
| 4 | `sparse_format_conversion/10000x10000_d0_csr_to_csc/10000` | `597.97 us` |
| 5 | `sparse_format_conversion/10000x10000_d0_csc_to_csr/10000` | `561.20 us` |
