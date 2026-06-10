# Rejected: single-pass parallel SpMM chunk assembly

Bead: `frankenscipy-8l8r1.79`

Target: `sparse_spmm/2000x2000_d1/2000`

Worker: RCH `vmi1227854`

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 CARGO_BUILD_JOBS=1 rch exec -- env CARGO_BUILD_JOBS=1 cargo bench -j 1 -p fsci-sparse --bench sparse_bench -- sparse_spmm/2000x2000_d1/2000 --warm-up-time 1 --measurement-time 5 --sample-size 30 --noplot
```

Criterion time: `[10.006 ms 10.672 ms 11.533 ms]`

Evidence: `baseline_spmm_2000_rch.txt`

## Candidate

Lever: remove the duplicate symbolic row-count traversal in each parallel chunk, let the numeric pass return per-row counts, and reserve chunk output by a coarse `a_chunk_nnz * avg_b_row` estimate.

Criterion time: `[16.395 ms 18.699 ms 21.141 ms]`

Delta by mean: `18.699 / 10.672 = 1.752x` slower.

Score: `0.0` because impact is negative; change was not kept.

Evidence: `after_spmm_2000_rch.txt`

## Behavior Proof

Focused proof:

```bash
RCH_REQUIRE_REMOTE=1 CARGO_BUILD_JOBS=1 rch exec -- env CARGO_BUILD_JOBS=1 cargo test -j 1 -p fsci-sparse --lib --locked spmm_parallel_matches_serial_byte_for_byte -- --nocapture --test-threads=1
```

Result: `1 passed; 0 failed`

Golden payload SHA256 before:

```text
0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2
```

Golden payload SHA256 after:

```text
0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2
```

Payload line counts: `141849` before, `141849` after.

Isomorphism notes:

- Row order stayed chunk/range order and per-row order stayed reverse first-seen order from `spmm_row_chunk`.
- Floating-point accumulation order within each row stayed unchanged.
- Zero elision stayed `v.abs() > 0.0`.
- `sorted_indices` metadata stayed the chunk-wise conjunction.
- RNG is not used.
- No unsafe code and no external BLAS/LAPACK linkage were introduced.

## Rejection Analysis

The removed counts pass also provided exact output capacity. Replacing it with a coarse average-row estimate preserved behavior but increased allocation growth in the hot numeric pass enough to lose badly on the profile-backed target.

Next deeper route: keep the algorithmic target on SpMM but attack allocation and symbolic reuse directly, such as row-local arena/slab reuse or a communication-avoiding symbolic plan cached across repeated structure-compatible products, with the same byte-for-byte golden proof.
