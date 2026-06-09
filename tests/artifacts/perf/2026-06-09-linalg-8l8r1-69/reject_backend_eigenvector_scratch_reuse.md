# frankenscipy-8l8r1.69: backend eigenvector scratch reuse rejected

## Target

- Bead: `frankenscipy-8l8r1.69`
- Profile-backed residual: square public `svd()` core after prior public-route keeps.
- Candidate: reuse one `v_col` scratch buffer while converting tridiagonal eigenvectors into bidiagonal SVD factors.

## Same-Worker Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 rch exec -- cargo test -p fsci-linalg --release --lib --locked deterministic_thin_svd_stage_breakdown_probe -- --ignored --nocapture --test-threads=1
```

Worker: `vmi1227854`

```text
reduction_ms=68.148
bidiagonal_svd_ms=52.852
back_transform_u_ms=14.686
back_transform_v_ms=20.440
```

## After

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_TEST_SLOTS=1 rch exec -- cargo test -p fsci-linalg --release --lib --locked deterministic_thin_svd_stage_breakdown_probe -- --ignored --nocapture --test-threads=1
```

Worker: `vmi1227854`

```text
reduction_ms=119.902
bidiagonal_svd_ms=67.933
back_transform_u_ms=40.081
back_transform_v_ms=57.955
```

## Isomorphism

The candidate copied the same eigenvector column entries in the same row order into the same `v_col` contents before calling `canonicalize_slice_sign`. `Vt` row writes and U reconstruction used the existing order and formulas. No ordering, tie, sign, floating-point arithmetic, RNG, rank, or fallback policy changed.

## Decision

Reject. Same-worker backend ratio was `52.852 / 67.933 = 0.777996x`; full stage ratio was `(68.148 + 52.852 + 14.686 + 20.440) / (119.902 + 67.933 + 40.081 + 57.955) = 0.511462x`. Source restored.

Next route: direct bidiagonal divide-and-conquer or dqds-style backend, or a communication-avoiding blocked reduction with a same-process A/B proof. Do not repeat backend allocation/indexing cleanup.
