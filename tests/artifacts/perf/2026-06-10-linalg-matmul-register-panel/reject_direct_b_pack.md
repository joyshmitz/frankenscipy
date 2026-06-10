# frankenscipy-cckrw direct B-pack rejection

Date: 2026-06-10
Bead: `frankenscipy-cckrw`
Worker: `vmi1227854`
Verdict: rejected, no source retained

## Lever

Remove the temporary row-major `b_flat` copy from `matmul_flat_workspace`, packing complete 8-column B panels directly from the input rows and loading scalar tails from `b[k][j]`.

The lever preserved the arithmetic contract: output order, tie behavior, RNG surface, row splitting, public dispatch, and per-output `k` accumulation order were unchanged. The final source was restored to baseline after the timing gate failed.

## Proofs

Baseline artifact integrity passed:

```text
baseline_matmul_criterion_rch.txt
sha256 62d9ffc74fa324d0f1fc8d26684227d0cd8fe63781c050eaca65969cc02fae89
```

Final RCH proofs on `vmi1227854` passed:

```text
proof_final_matmul_debug_group_rch.txt
sha256 7cdc033feccb96ea7ac56b4c6bcb5b6ab302ab73989a0f6c0376ae4bdf417ceb
result 5 matmul tests passed, 0 failed

proof_final_matmul_medium_flat_workspace_golden_rch.txt
sha256 c52f716411e9490c174026353374e4787885c1f0deb6c864bba77839ba755783
digest 0x5fd37bf053d54fb0
result passed
```

## Benchmarks

Original clean baseline:

```text
baseline_matmul_criterion_rch.txt
sha256 62d9ffc74fa324d0f1fc8d26684227d0cd8fe63781c050eaca65969cc02fae89
256  4.9797 ms
512 31.6110 ms
768 91.0720 ms
1024 281.0400 ms
```

The first candidate run was noisy: it improved 768/1024 but regressed the untouched 256 sentinel, so it was not used as keep evidence.

```text
candidate_matmul_direct_b_pack_criterion_rch.txt
sha256 b70ec54a593b687d7f5770c09c18f3d11ed325254c042991fb01a4d70b3fbefa
256  5.1781 ms
512 31.4240 ms
768 87.3090 ms
1024 145.3200 ms
```

Paired restored-baseline rerun:

```text
paired_baseline_bflat_criterion_rch.txt
sha256 3d9903e6d25bf74c27bacb43c0808af53e18d618ca9a280cbf89b5d19723c50c
256  5.3600 ms
512 39.3090 ms
768 123.6000 ms
1024 239.5500 ms
```

The clippy-safe final candidate regressed against the paired restored baseline:

```text
final_candidate_direct_b_pack_criterion_rch.txt
sha256 513735e8f6be2717fa09de6a801beb412e357423ed832198d3e797de985140f8
256  5.6784 ms  ratio 0.943928x
512 41.8300 ms  ratio 0.939732x
768 154.4500 ms ratio 0.800259x
1024 275.5200 ms ratio 0.869447x
affected geomean ratio 0.867946x
```

Score: below zero effective impact after final candidate timing; reject.

## Validation Notes

`cargo fmt -p fsci-linalg --check` passed. RCH `cargo check -p fsci-linalg --all-targets --locked` passed before the final source restore. Full `cargo clippy -p fsci-linalg --all-targets --locked` was blocked by an unrelated `fsci-fft` dependency lint. Scoped no-deps clippy exposed a candidate packing-loop lint; after fixing it, the final candidate benchmark regressed and the source was restored, so no source clippy state remains to validate.

## Route

Do not repeat redundant B-copy removal or direct-pack variants. The measured residual is still GEMM compute, not staging. Next profile-backed route should attack a deeper primitive: register-panel/thread scheduling stability, row-panel cache layout, or a new safe-Rust blocked GEMM microkernel with an explicit same-worker A/B probe that isolates compute from RCH load variance.
