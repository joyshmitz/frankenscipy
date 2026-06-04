# Sparse COO-to-CSR Duplicate Sort Evidence

Bead: `frankenscipy-ebuyh`

Target: `sparse_csr_construction/10000x10000_d0/10000`, selected from the post-diags sparse reprofile where it was the top non-owned sparse hotspot at median `5.0004 ms`.

## One Lever

`CooMatrix::to_csr` no longer sorts by `(row, col)` after `canonical_triplets(self)`. The helper already returns triplets sorted by `(row, col)` and deduplicated, so the second sort was redundant for CSR compression.

## Benchmarks

All benchmark commands were crate-scoped and run through RCH.

- Baseline: `cargo bench -p fsci-sparse --bench sparse_bench --locked -- sparse_csr_construction/10000x10000_d0/10000 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot`
  - Worker: `ts2`
  - Median: `4.8411 ms`
- After: same command
  - Worker: `ts2`
  - Median: `4.5528 ms`
- After confirm: same command
  - Worker: `ts2`
  - Median: `4.6274 ms`

Conservative confirmed speedup: `4.8411 ms / 4.6274 ms = 1.047x`.

Score: `Impact 1.047 * Confidence 0.95 / Effort 0.4 = 2.49`, keep.

## Isomorphism Proof

- `canonical_triplets` is still called first, preserving duplicate merge order and row-major `(row, col)` order.
- CSR compression still receives row-major sorted triplets and builds the same `indptr`, `indices`, and `data`.
- Explicit zero entries and duplicate sums are preserved; no zero elision was introduced.
- Floating-point values are copied or summed only in `canonical_triplets`, unchanged by this lever.
- No RNG behavior changes in library code; the seeded golden fixture is only an output witness.
- `to_csc` still performs its column-major sort because `canonical_triplets` is row-major.

Golden output before and after:

- `golden_before.txt` SHA256: `943927e5ee49288577e3ed37e13b8f38c76aec8d0b71ac159b4895905afd6df1`
- `golden_after.txt` SHA256: `943927e5ee49288577e3ed37e13b8f38c76aec8d0b71ac159b4895905afd6df1`
- `golden_before_after.diff`: empty

## Validation

- `cargo fmt -p fsci-sparse --check`: pass
- `RCH_FORCE_REMOTE=1 rch exec -- cargo check -p fsci-sparse --all-targets --locked`: pass
- `RCH_FORCE_REMOTE=1 rch exec -- cargo test -p fsci-sparse --lib --locked`: pass, `307 passed`
- `RCH_FORCE_REMOTE=1 rch exec -- cargo clippy -p fsci-sparse --all-targets --locked -- -D warnings`: pass
- `ubs --ci --only=rust crates/fsci-sparse/src/ops.rs crates/fsci-sparse/src/bin/perf_sparse.rs`: exit 0
