# Reject: guarded normal-equation thin-SVD route

Bead: `frankenscipy-8l8r1.59`
Date: 2026-06-08

## Lever

Add a guarded public tall thin-SVD candidate that forms `A^T A`, decomposes the Gram matrix, reconstructs `U = A V Sigma^-1`, and falls back to the existing bidiagonal route when gates reject.

## Proof

The focused candidate proof passed on a 128x64 deterministic input:

```text
test tests::public_gram_thin_svd_candidate_matches_safe_svd_reference ... ok
test tests::public_gram_thin_svd_candidate_respects_size_gate ... ok
```

## Baseline

Persisted baseline artifact:
`baseline_public_bidiag_svd_route_perf_probe_rch.txt`

```text
worker=vmi1167313
routed_lstsq_ms=96.299490
routed_pinv_ms=145.510990
lstsq_max_abs_diff=1.07647224467655178e-12
pinv_max_abs_diff=2.28428387316625958e-14
```

## After

The same public-route probe failed after the candidate was enabled:

```text
routed_lstsq_ms=246.792920
routed_pinv_ms=319.768046
lstsq_max_abs_diff=3.55636732507491615e-7
pinv_max_abs_diff=1.52151365395702953e-8
failure=assertion failed: lstsq_max_abs_diff <= 1e-7
```

The route was slower and violated the existing `lstsq` tolerance contract on the profile target.

## Verdict

Rejected and source restored. Do not repeat the normal-equation/Gram shortcut as a public route. Next primitive: blocked one-sided Jacobi thin-SVD that works on `A` directly, avoids condition-number squaring, and remains fail-closed behind public reconstruction/rank/tie/sign gates.
