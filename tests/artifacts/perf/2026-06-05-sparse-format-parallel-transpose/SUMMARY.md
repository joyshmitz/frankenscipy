# frankenscipy-8l8r1.33 - Parallel Direct Compressed Transpose Trial

## Target

- Bead: `frankenscipy-8l8r1.33`
- Profile-backed rows:
  - `sparse_format_conversion/10000x10000_d0_csr_to_csc/10000`
  - `sparse_format_conversion/10000x10000_d0_csc_to_csr/10000`

## Lever Tried

Tried a safe-Rust parallel segmented-histogram transpose for canonical compressed CSR/CSC inputs. Each worker built a local compressed output chunk for a contiguous major-axis range; the final output concatenated each output segment in worker order to preserve sorted row/column order.

The source change was rejected and manually restored. `crates/fsci-sparse/src/ops.rs` has no remaining diff for this trial.

## Behavior Proof

- Existing structural guard `can_direct_transpose_compressed` was preserved.
- Fallback behavior for noncanonical or mislabelled input was unchanged.
- Output order was preserved by contiguous major ranges and per-output-segment worker-order concatenation.
- Data order was preserved within each output segment by scanning each worker range in original major order.
- Metadata was unchanged: output stayed `sorted_indices=true`, `deduplicated=true`.
- Floating-point values were copied, not recomputed; no rounding changes.
- RNG and tie-breaking do not participate.
- RCH `conversion_golden_snapshot` passed.
- Golden SHA-256 stayed `f01e261f50d39eab13c364c8af2dee85d335ad78729e18d2014dfa17450d2efe`.
- Golden payload compare against the accepted direct-transpose artifact exited `0`.

## Benchmarks

Focused same-worker RCH baseline on `ts1`:

- CSR -> CSC: `545.15 us` median `[536.74, 553.86]`
- CSC -> CSR: `543.21 us` median `[535.93, 549.16]`

After on `ts1`:

- CSR -> CSC: `2.4028 ms` median `[2.3758, 2.4297]`
- CSC -> CSR: `2.3583 ms` median `[2.3365, 2.3815]`

## Verdict

Rejected. Score `0.0`, below the keep threshold. The extra per-worker materialization plus final segment concatenation outweighed parallelism at the profiled 100k-nnz conversion size.
