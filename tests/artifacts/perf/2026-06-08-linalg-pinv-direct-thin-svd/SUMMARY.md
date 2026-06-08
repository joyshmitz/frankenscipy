# Direct Thin-SVD Pseudoinverse Assembly

Bead: `frankenscipy-8l8r1.49`

Decision: rejected and source restored.

## Target

Profile-backed row: `pinv/512x256` in `fsci-linalg`.

Primitive attempted: direct safe-Rust row assembly for the accepted deterministic thin-SVD `pinv` route, avoiding `DMatrix` pseudoinverse materialization.

## Baseline

Command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- pinv/512x256 --warm-up-time 1 --measurement-time 5 --sample-size 30 --noplot
```

Worker: `fmd`

Criterion: `[51.098 ms 51.694 ms 52.261 ms]`

Artifact: `baseline_pinv_512x256_rch.txt`

## Proof

Focused direct-reference proof passed on `fmd` before restoration:

```bash
RCH_FORCE_REMOTE=1 rch exec -- cargo test -p fsci-linalg --release --lib public_bidiag_svd_direct_pinv_matches_dmatrix_reference --locked -- --nocapture
```

Artifact: `proof_direct_pinv_reference_rch.txt`

Public golden SHA-256 before:

```text
1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225
```

Public golden SHA-256 after:

```text
1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225
```

Artifacts:

- `golden_public_before_rch.txt`
- `golden_public_before_payload.sha256`
- `golden_public_after_rch.txt`
- `golden_public_after_payload.sha256`

Isomorphism notes: public small golden output was byte-identical. The direct helper matched the old matrix-product helper within the existing linalg tolerance policy on the focused large deterministic case. No RNG, ordering, rank, route gate, threshold, or certificate logic was changed in the attempted patch.

## Re-benchmark

Unpinned after-run, not used for the formal gate:

- Worker: `vmi1293453`
- Criterion: `[117.00 ms 119.02 ms 121.15 ms]`
- Artifact: `after_pinv_512x256_rch.txt`

Pinned same-worker gate:

```bash
RCH_FORCE_REMOTE=1 RCH_WORKER=fmd rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- pinv/512x256 --warm-up-time 1 --measurement-time 5 --sample-size 30 --noplot
```

- Worker: `fmd`
- Criterion: `[107.67 ms 108.24 ms 108.89 ms]`
- Artifact: `after_pinv_512x256_fmd_rch.txt`

Same-worker ratio: `51.694 / 108.24 = 0.477x`.

## Score

Score: rejected, below the `>= 2.0` keep gate. The lever improved no measured target and regressed same-worker runtime by `2.09x`.

## Follow-up Route

Do not repeat this direct-final-materialization family. Reprofile the restored source and move to a structurally different primitive that reduces the SVD/GEMM work itself, such as blocked bidiagonal update fusion, communication-avoiding SVD stages, or a deeper packed-kernel primitive.
