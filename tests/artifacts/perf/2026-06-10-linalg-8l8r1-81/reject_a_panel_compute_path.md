# Rejected: A-Panel GEMM Compute Path

Date: 2026-06-10
Bead: `frankenscipy-8l8r1.81`
Lever: per-row-chunk K-major A panel for flat-workspace GEMM compute.
Verdict: rejected; no source retained.

## Why This Was Tried

The previous `frankenscipy-cckrw` direct B-pack lever was proof-clean but slower on
same-worker Criterion evidence. This pass therefore avoided B staging and attacked
the compute path directly by changing A-side row-panel cache layout. The arithmetic
contract remained simple: each output cell still accumulates `k = 0..ka` in increasing
order, and the A-panel invariant was checked in the direct row-split proof.

## Behavior Proof

All behavior proof commands completed before benchmarking the candidate:

- Baseline SHA verification: passed.
- `matmul_flat_workspace_is_bit_identical_to_naive_ijk`: passed on `ovh-a`.
- `matmul_flat_compute_rows_row_split_is_bit_identical`: passed on `ovh-a`, including K-major A-panel invariant checks.
- `matmul_microkernel_golden_digest`: passed on `ovh-a`.
- Release ignored `matmul_medium_flat_workspace_route_golden_digest`: passed on `ovh-a`, digest `0x5fd37bf053d54fb0`.

Quality gates:

- `cargo fmt -p fsci-linalg --check`: passed while the candidate source was present.
- `ubs crates/fsci-linalg/src/lib.rs`: exit 0, 0 critical findings.
- RCH `cargo check -j 1 -p fsci-linalg --all-targets --locked`: passed on `ovh-a`.
- Clippy remains blocked by pre-existing unrelated lints:
  - dependency `fsci-fft` `manual_is_multiple_of`;
  - `fsci-linalg` non-GEMM `manual_memcpy` at `src/lib.rs:4231`.

## Same-Worker Benchmark

Candidate and paired restored baseline both ran on RCH worker `vmi1227854`.

| Benchmark | Paired baseline mean | Candidate mean | Speed ratio |
| --- | ---: | ---: | ---: |
| `matmul/512x512` | `8.8947 ms` | `14.883 ms` | `0.597642x` |
| `matmul/768x768` | `29.300 ms` | `33.870 ms` | `0.865072x` |
| `matmul/1024x1024` | `70.485 ms` | `185.06 ms` | `0.380876x` |

Affected-size geomean: `0.581780x`.

The 256 sentinel also moved only from `5.4766 ms` to `5.1824 ms`, so the affected-size
regression is not a harmless worker-speed artifact.

## Score

Impact is negative, so Score is below the `>= 2.0` keep gate. Source was restored to
zero diff.

## Do Not Repeat

Do not repeat:

- direct B staging or B direct-pack variants;
- per-worker K-major A row-slab paneling in the current `MR=4`, `NR=8`, `NC=16` kernel;
- row-panel packing variants that add per-chunk A repacking without replacing the
  macro-kernel structure.

Next route: a materially different BLIS-style safe-Rust GEMM macro-kernel from the
communication-avoiding/polyhedral locality family: explicit `MC/KC/NC` blocking with
reused packed panels and a wider register-panel microkernel, or an `NC=32` multi-panel
kernel if the profile first isolates register pressure and cache reuse. Fresh RCH
baseline is required before the next source lever.
