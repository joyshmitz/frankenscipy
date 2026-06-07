# GEMM Row-Split Granularity Rejection

Bead: `frankenscipy-8l8r1.40`

## Profile Target

Post-`frankenscipy-vgs9h` crate-scoped RCH profile on `ts1`:

```text
command: RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- --warm-up-time 1 --measurement-time 3 'matmul/1024x1024|lstsq/512x256|pinv/512x256|solve/256x256|inv/256x256|det/256x256' --noplot
solve/256x256     [3.5533 ms 3.5964 ms 3.6433 ms]
inv/256x256       [12.896 ms 13.050 ms 13.217 ms]
det/256x256       [1.4366 ms 1.4474 ms 1.4588 ms]
lstsq/512x256     [71.687 ms 72.486 ms 73.381 ms]
pinv/512x256      [75.996 ms 77.021 ms 78.175 ms]
matmul/1024x1024  [53.361 ms 57.438 ms 61.219 ms]
```

## Lever Tested

Change only `matmul_thread_count` so the existing exact-order parallel flat-workspace GEMM caps workers at one thread per 32 output rows instead of one per 64 rows.

This preserved the arithmetic contract: each output `c[i][j]` still accumulates `k` in monotonic `0..ka` order, and only row ownership changed.

## Proof

RCH focused proof on `vmi1227854`:

```text
command: RCH_FORCE_REMOTE=1 rch exec -- cargo test -p fsci-linalg --release --lib matmul --locked -- --nocapture
result: ok
passed: matmul_ikj_is_bit_identical_to_naive_ijk
passed: matmul_flat_compute_rows_row_split_is_bit_identical
passed: matmul_flat_workspace_is_bit_identical_to_naive_ijk
passed: matmul_microkernel_is_bit_identical_to_flat_ikj
passed: matmul_microkernel_golden_digest
```

Isomorphism:

- Ordering: output row order and column order unchanged.
- Tie-breaking: not applicable.
- Floating point: every cell keeps the same monotonic `k` accumulation; proof compares `f64::to_bits`.
- RNG: none.
- Golden output: `matmul_microkernel_golden_digest` stayed unchanged.

## Rebench

RCH after-run on `vmi1227854`:

```text
command: RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- --warm-up-time 1 --measurement-time 3 'matmul/1024x1024' --noplot
matmul/1024x1024 [104.14 ms 113.80 ms 132.85 ms]
```

This was not same-worker against the `ts1` baseline, but it was a bad enough first signal to reject the scheduling-granularity family. Source was restored to the 64-row cap.

## Decision

Rejected. Score `0.0`.

Next primitive: deeper exact-order GEMM reuse, such as a wider register tile or a more structural packed-panel microkernel. Do not continue worker-count or row-scheduling micro-levers for this target.
