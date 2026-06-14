# frankenscipy-8l8r1.111 matmul B-panel traversal reject

Agent: RubyWaterfall
Date: 2026-06-14
Target: `crates/fsci-linalg/src/lib.rs::matmul_flat_compute_rows`

## Candidate

Tested one lever: move the `j0`/B-panel loop outside the `ib` row-block loop inside each thread-owned row range. The candidate preserved output-cell ownership and the per-cell `k` accumulation order, but changed cache traversal by reusing each B-panel across all row blocks in the worker chunk.

## Proof

- `rch exec -- cargo test -p fsci-linalg --lib --release matmul_flat -- --nocapture` passed.
- `rch exec -- cargo test -p fsci-linalg --lib --release matmul_medium_flat_workspace_route_golden_digest -- --ignored --nocapture` passed.
- Golden digest remained `matmul_medium_flat_route_golden_digest=0x6e401fad043ac8fd`.
- `rustfmt --edition 2024 --check crates/fsci-linalg/src/lib.rs` passed.
- After manual source restore, `git diff --exit-code -- crates/fsci-linalg/src/lib.rs` passed.
- Restored `crates/fsci-linalg/src/lib.rs` SHA-256: `e6d47a43cda59df5193b42fc6daff4d93afdffe1026969496c1b8b954be48624`.

## Same-worker RCH benchmark

Worker: `vmi1227854`

| row | baseline p50 | candidate p50 | ratio |
| --- | ---: | ---: | ---: |
| `matmul/256x256` | `3.1882 ms` | `3.3415 ms` | `0.954x` |
| `matmul/512x512` | `27.696 ms` | `19.479 ms` | `1.422x` |
| `matmul/768x768` | `92.150 ms` | `145.44 ms` | `0.634x` |
| `matmul/1024x1024` | `173.38 ms` | `99.678 ms` | `1.739x` |

The 1024x1024 improvement is not shippable because the 768x768 sentinel regressed by about 58% p50. A targeted 768x768 confirmation attempt failed due remote convergence noise: worker `vmi1227854` reported missing tracked local file `crates/fsci-linalg/src/bin/diff_signm.rs`.

## Verdict

Score: `0.0`

Reject. Do not retry simple B-panel traversal spelling, direct B-copy/staging/direct-pack, panel-load spelling, scalar-splat spelling, A row-slab packing, RB=128 row-panel geometry, KC-striped prefix writeback, row-owned materialization, row-split granularity, threshold retunes, or MR/NR widening.

Next linalg matmul pass should attack a different primitive: recursive/cache-oblivious square GEMM split, register-blocked microkernel schedule with explicit edge kernels, or a mixed precision/refinement artifact only where the tolerance contract can be proven unchanged.
