# frankenscipy-8l8r1.74 GEMM Row-Split Rejection

## Target

- Bead: `frankenscipy-8l8r1.74`
- Profile target: `fsci-linalg` public `matmul/1024x1024`
- Lever attempted: raise the flat-workspace row-split minimum from 64 rows per worker to 128 rows per worker.

This row-split attempt was originally created locally as `.73`; it was renumbered to `.74` during rebase because upstream had already opened `.73` for the dense-eigh blocked-reduction route.

## Baseline

- Command: `RCH_WORKER=ovh-a rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- matmul`
- Worker selected: `ovh-a`
- Source setting: `matmul_thread_count` cap `m / 64`
- `matmul/1024x1024`: `[51.845 ms 52.391 ms 52.986 ms]`
- Artifact: `baseline64_matmul_criterion_rch.txt`

## After

- Command: `RCH_WORKER=vmi1227854 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- matmul`
- Worker selected: `ovh-a`
- Source setting: `matmul_thread_count` cap `m / 128`
- `matmul/1024x1024`: `[63.152 ms 63.736 ms 64.294 ms]`
- Artifact: `after_matmul_rowsplit_criterion_rch.txt`

## Decision

- Same-worker 1024 ratio: `52.391 / 63.736 = 0.8220x`
- Score: below `2.0`; reject.
- Source retained: none. The source tree was restored to the baseline `m / 64` row split.

## Isomorphism

- Ordering preserved: yes; GEMM writes the same output coordinates.
- Tie-breaking unchanged: not applicable.
- Floating-point: each output element still accumulates `k` monotonically in `0..ka`; only row partitioning was attempted.
- RNG seeds: not applicable.
- Golden outputs: no source retained; existing GEMM golden route remains unchanged.

## Route Forward

Do not repeat row-split granularity as the next GEMM lever. The next pass should attack a different profile-backed primitive, such as deeper packed-panel/register blocking or an algorithmic dense linear algebra route from the no-gaps directive.
