# SpGEMM (spmm): parallelize across independent output rows

Follow-up to the dense-accumulator commit (a7412152). Function: `linalg::spmm`.

## Lever
Each output row is an independent Gustavson merge, so for heavy products the rows
are fanned out across a `std::thread::scope` pool (the project's parallelism
convention — same shape as the parallel CSR add). Each worker owns a private
dense accumulator (`acc`+`seen`, length n) and emits its rows into chunk-local
buffers; the driver concatenates them in row order and rebuilds indptr from the
per-row counts. Gated by an estimated-flops proxy (`nnz(A) * avg nnz per B row`),
since SpGEMM fill grows superlinearly and `nnz(A)` alone is a poor cost signal —
only products ≥ ~300k multiply-adds (and ≥512 rows) parallelize.

## Isomorphism
A row's columns/values depend only on that row; the reverse first-seen emit order
is per-row; `sorted_indices` is an associative AND across rows. So chunked,
row-ordered concatenation is byte-identical to the serial sweep for any worker
count.
- `spmm_parallel_matches_serial_byte_for_byte`: asserts parallel == single-chunk
  serial (cols/vals/indptr/sorted) for worker counts {2,3,7,8,16} on an 800² product.
- Golden SHA over the spmm payload unchanged:
  `0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2`
  (serial == HEAD; equivalence test extends that to the parallel path).
- 309 lib tests pass; clippy clean.

## Benchmark (rch ts2, same-worker A/B: forced-serial vs parallel)
| case          | serial     | parallel  | Score |
|---------------|------------|-----------|-------|
| 2000x2000 d1% | 29.736 ms  | 9.873 ms  | 3.01x |

Smaller configs stay serial by the flop gate (500x500 d2% ~0.57ms, 1000x1000 d1%
~1.17ms — below the parallel threshold; per-call spawn does not amortise there).
Combined with the dense-accumulator commit, 2000x2000 SpGEMM is now bounded by
real parallel compute instead of HashMap churn.
