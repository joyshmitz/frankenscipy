# SpMM Work-Proportional Output Capacity

Bead: `frankenscipy-1zaob`
Crate: `fsci-sparse`
File: `crates/fsci-sparse/src/linalg.rs`

## Target

`sparse_spmm/2000x2000_d1/2000` stayed the dominant sparse hotspot after the
adaptive row-band trial was rejected. The existing parallel Gustavson SpMM path
gave every worker an output buffer capacity based on `nnz(A)` per chunk, which
underestimated output for random 1% x 1% products and forced repeated `Vec`
growth.

## Lever Tried

The trial sized each worker's output buffers from estimated multiply-add work:

`nnz(A) * ceil(nnz(B) / rows(B)) / chunk_count`

The row ranges, row-local accumulator, first-seen column list, reverse emission
order, and final row-order chunk concatenation are unchanged.

## Isomorphism

- Ordering preserved: yes; row ranges and chunk concatenation are unchanged.
- Tie-breaking unchanged: yes; per-row first-seen column order is unchanged.
- Floating-point unchanged: yes; only allocation capacity changes, not product
  encounter or addition order.
- RNG unchanged: yes; no runtime RNG is used by `spmm`.
- Strict golden SHA-256 unchanged:
  `0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2`

The strict before/after golden payload diff is empty.

## Performance

Initial baseline on `vmi1153651`:

- baseline: `[18.751 ms, 20.208 ms, 21.859 ms]`
- after confirmation: `[24.843 ms, 28.088 ms, 31.954 ms]`

Paired side evidence:

- `ts2`: `12.787 ms` -> `11.990 ms` median
- `ts1`: `9.6832 ms` -> `9.4374 ms` median, but intervals overlap

The mixed-worker evidence is not strong enough to keep the lever. The same
original worker regressed badly, so the source change was restored.

Score: `0.0`; rejected.

## Validation

- `cargo fmt -p fsci-sparse --check`: passed after restore
- `git diff --quiet -- crates/fsci-sparse/src/linalg.rs`: exit 0 after restore

Next pass should reprofile and pivot to a deeper SpGEMM primitive, not another
capacity tweak: row-work prefix partitioning or a CSC/column-panel GraphBLAS
traversal with the same row-local floating-point/order contract.
