# frankenscipy-8l8r1.81 splat-reuse panel rejection

Date: 2026-06-10
Agent: BlackThrush
Bead: `frankenscipy-8l8r1.81`
Verdict: rejected, no source retained

## Lever

Reuse `Simd::splat(a*)` values across the two adjacent eight-lane B vectors in
the full 4x16 GEMM tile instead of spelling out a fresh splat for each half
panel.

This was a compute-path-only probe. It did not change dispatch thresholds,
row splitting, B packing, output ordering, RNG surface, tie behavior, or the
per-output monotonic `k` accumulation order. The source was restored after the
timing gate failed.

## Proofs

```text
proof_splat_reuse_matmul_group_rch.txt
sha256 2c1bfdee08da386e03b544ebf59b5d401628b3883d8091b68f7d2b1f03e38ad2
worker ovh-a
result 5 matmul tests passed, 0 failed

proof_splat_reuse_medium_flat_workspace_golden_rch.txt
sha256 43182a0f14eed2dbe45ad37202ce314d039efa7553aad15d37aab98362b697f6
worker ovh-a
digest 0x5fd37bf053d54fb0
result passed
```

## Benchmarks

The first after-run refused local fallback because no admissible remote worker
was available:

```text
after_splat_reuse_matmul_criterion_rch.txt
sha256 56c01fa0e88b07525727b26ef3de59efe189ec9645b3b53c7372a70cef3a9742
result RCH remote required; refusing local fallback
```

The retry selected `vmi1227854`, so it is routing evidence rather than a valid
same-worker comparison to the `ovh-a` baseline:

```text
after_splat_reuse_matmul_criterion_rch_retry1.txt
sha256 e24fdd8c3b82f5c880b280f21b72bfaa3c429cda821e5ca7091e3a30defa2a73
worker vmi1227854
256   5.7776 ms
512  45.5330 ms
768  92.6470 ms
1024 175.8900 ms
```

Against the nearest same-worker restored-baseline artifact
`paired_baseline_bflat_criterion_rch.txt`, the routing ratios were mixed:

```text
256  ratio 0.927721x
512  ratio 0.863308x
768  ratio 1.334096x
1024 ratio 1.361931x
affected geomean ratio 1.161901x
all-size geomean ratio 1.098326x
```

Score: below the keep threshold. The 512 profile-backed size regressed and the
available signal is not a valid same-worker keep pair. Reject.

## Route

Do not repeat scalar-splat spelling variants. The residual needs a deeper
compute primitive: register-tile shape, row-panel cache layout, or a
communication-avoiding blocked GEMM path with isolated same-worker A/B proof.
