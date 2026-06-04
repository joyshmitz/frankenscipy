# SpGEMM (spmm): HashMap row accumulator → dense Gustavson SPA

Bench: `sparse_spmm` (fsci-sparse), rch ts2. Function: `linalg::spmm` (CSR × CSR).

## Lever
The per-row product accumulator was a fresh `HashMap<usize, f64>` allocated for
every output row (hash cost on every product term + an allocation/realloc per
row). Replaced with a dense **sparse accumulator (SPA / Gustavson)**: an
`acc: Vec<f64>` and `seen: Vec<bool>` of length `n` (= B columns) allocated once
and reused across all rows, cleared only at the columns each row touches (via the
existing `column_order` list). Array reads/writes are O(1) with no hashing and no
per-row allocation.

## Isomorphism
Each product `a_ik * b_kj` is added into `acc[j]` in the exact encounter order
the HashMap used, so accumulated values are bit-identical; the reverse-first-seen
emit order (SciPy CSR-matmul parity), zero elision, and sorted/dedup metadata are
all unchanged.
- Golden SHA over spmm payload (indptr+indices+value bits, two random products
  500²@2% and 1000²@1%): `0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2`
  — identical with the new dense accumulator and the HEAD HashMap version.
- 308 lib tests pass (incl. `spmm_identity_*` parity tests); clippy clean.

## Benchmark (rch ts2)
| case            | before    | after     | Score |
|-----------------|-----------|-----------|-------|
| 500x500 d2%     | 5.0065 ms | 0.5398 ms | 9.27x |
| 1000x1000 d1%   | 10.298 ms | 1.1061 ms | 9.31x |

Both ~9.3x, well past the Score>=2.0 bar.

## Note (separate bead, vifqu / CSR add)
The CSR *add* hot path (`ops::combine_csr_rows_directly`) is already an optimal
sorted two-way merge; a dense SPA accumulator there would be slower (clearing an
n-length array per 10-nnz row). Per-call `std::thread` parallelism of that merge
caps at ~1.43x (spawn + a serial concat that safe Rust cannot avoid without
double-merging or `set_len`). Branchless merge gave 0x (write-bound, not
branch-bound). That bead needs a persistent thread pool or a heavier target — the
SPA primitive belongs in SpGEMM (this commit), not SpAdd.
