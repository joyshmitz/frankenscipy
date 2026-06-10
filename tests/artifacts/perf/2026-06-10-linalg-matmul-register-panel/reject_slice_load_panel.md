# frankenscipy-8l8r1.81 slice-load panel rejection

Date: 2026-06-10
Agent: BlackThrush
Bead: `frankenscipy-8l8r1.81`
Worker: `ovh-a`
Verdict: rejected, no source retained

## Lever

Replace the hand-expanded eight-lane packed-B panel loads in the 4x16 full-tile
GEMM path with `Simd::from_slice(&packed_b[base..base + NR])`.

This was a narrow compute-path probe only. It did not change dispatch, row
partitioning, B packing, output ordering, RNG surface, tie behavior, or the
per-output monotonic `k` accumulation order. The source was restored after the
timing gate failed.

## Proofs

```text
proof_slice_load_flat_workspace_bits_rch.txt
sha256 207f4a65afcdd7911da4b20de29ad04f9dfc39a1f133aa20cfdb068026ac44e4
result matmul_flat_workspace_is_bit_identical_to_naive_ijk passed

proof_slice_load_matmul_group_rch.txt
sha256 19a639fd2983869231eaa75bf2c1096c52e06aea8a8beab6006dfea707bfc01c
result 5 matmul tests passed, 0 failed

proof_slice_load_medium_flat_workspace_golden_rch.txt
sha256 d07047db8afc5d6e16d8be9bb9aa621b908f0e33bc74e0a6d6b98b84465774e4
digest 0x5fd37bf053d54fb0
result passed
```

## Benchmarks

Baseline:

```text
baseline_matmul_criterion_rch_blackthrush_20260610.txt
sha256 a55d15485525ef8cbee2a09eb7d6951fc31797fee0465d2adad7da500c5dd2d0
256   4.9586 ms
512  11.1610 ms
768  29.0070 ms
1024 59.4290 ms
```

Candidate:

```text
after_slice_load_matmul_criterion_rch.txt
sha256 ddc6c155a7adf935e44c5e0ab85bc8b7b543bb39647e7de45edccde4bc76fd38
256   4.7564 ms  ratio 1.042510x
512  25.4970 ms  ratio 0.437738x
768  38.1030 ms  ratio 0.761279x
1024 78.8350 ms  ratio 0.753840x
affected geomean ratio 0.630975x
```

Score: below zero effective impact because the profile-backed 512/768/1024
cases regressed. Reject.

## Route

Do not repeat panel-load abstraction variants unless a later disassembly/profile
shows a different codegen path. Continue on a compute-path primitive that keeps
the current B-pack/staging route intact.
