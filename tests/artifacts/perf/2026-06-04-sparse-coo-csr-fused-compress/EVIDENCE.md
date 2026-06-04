# Sparse COO to CSR fused compression trial

Bead: `frankenscipy-ccm7m`
Date: 2026-06-04
Worker: RCH `vmi1149989`
Target: `sparse_csr_construction/10000x10000_d0/10000`

## Profile target

`br ready --json` was empty, so this pass used the fresh RCH sparse reprofile from
`tests/artifacts/perf/2026-06-04-sparse-reprofile-after-direct-transpose/reprofile_sparse_bench_rch.txt`.
After the CSR-add bookkeeping family failed in Pass 4, the next non-add sparse hotspot was
COO to CSR construction.

Alien primitive source:
- segmented sparse construction / GraphBLAS-style compressed formats
- loop fusion over canonicalization and compression
- sorted-array sparse access patterns

## Lever

Trialed one source lever in `crates/fsci-sparse/src/ops.rs`:
- keep the existing `sort_unstable_by_key(|(row, col, _)| (row, col))`;
- preserve duplicate accumulation order by merging only adjacent equal `(row, col)` entries after that same sort;
- emit CSR `data`, `indices`, and `indptr` during the dedup pass instead of materializing canonical triplets and walking them again.

The trial intentionally left `to_csc` unchanged and kept `CsrMatrix::from_components(..., true)` validation in place.

## Isomorphism proof

Command:
`RCH_WORKER=vmi1149989 rch exec -- cargo run -p fsci-sparse --bin perf_sparse --locked -- coo-csr-golden`

Golden before:
`943927e5ee49288577e3ed37e13b8f38c76aec8d0b71ac159b4895905afd6df1`

Golden after:
`943927e5ee49288577e3ed37e13b8f38c76aec8d0b71ac159b4895905afd6df1`

`golden_cmp.txt` records `golden_cmp_exit=0`.

Preserved behavior:
- row/column output order unchanged;
- duplicate floating-point accumulation order unchanged for the captured golden cases;
- explicit zero storage unchanged;
- no RNG, tie-breaking, validation-mode, or error-surface changes were kept.

## Benchmark gate

Command:
`RCH_WORKER=vmi1149989 rch exec -- cargo bench -p fsci-sparse --bench sparse_bench --locked sparse_csr_construction/10000x10000_d0/10000 -- --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot`

Baseline:
`[377.74 us 401.44 us 455.18 us]`

After:
`[545.03 us 588.65 us 679.45 us]`

Result: `0.682x` by median, a regression.

Score: `0.0 = impact 0 * confidence 4 / effort 2`.

Verdict: rejected. Source restored.

## Restore proof

After removing the lever:
- `git diff --quiet -- crates/fsci-sparse/src/ops.rs` exited `0`;
- `cargo fmt -p fsci-sparse --check` exited `0`.

## Next primitive

Do not retry this fused canonical-triplet compression family. The next sparse pass should pivot to a different profile-backed primitive, preferably a closed-form diagonal/tridiagonal constructor or a row-bucketed CSR construction algorithm with a separate proof that duplicate floating-point order and validation behavior are bit-identical.
