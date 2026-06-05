# Sparse Reprofile After frankenscipy-0uon5 Rejection

## Command

`RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-sparse --bench sparse_bench --locked -- --warm-up-time 1 --measurement-time 5 --sample-size 30 --noplot`

## Worker

RCH worker: `ts2`

## Top Rows

- `sparse_spmm/2000x2000_d1/2000`: `12.481 ms` median `[12.194, 12.709]`
- `sparse_arithmetic/10000x10000_d0_add/10000`: `1.6948 ms` median `[1.6143, 1.7727]`
- `sparse_spmm/1000x1000_d1/1000`: `1.0667 ms` median `[1.0633, 1.0706]`
- `sparse_format_conversion/10000x10000_d0_csc_to_csr/10000`: `824.71 us` median `[823.18, 827.60]`
- `sparse_format_conversion/10000x10000_d0_csr_to_csc/10000`: `820.61 us` median `[818.85, 822.39]`
- `sparse_csr_construction/10000x10000_d0/10000`: `621.14 us` median `[619.15, 623.47]`

## Selection

Created `frankenscipy-8l8r1.33` for a different profile-backed primitive: parallel direct compressed transpose for canonical CSR/CSC format conversion. This deliberately avoided another SpMM row/count/panel variant and another CSR-add bookkeeping/prefix/thread-cap variant.
