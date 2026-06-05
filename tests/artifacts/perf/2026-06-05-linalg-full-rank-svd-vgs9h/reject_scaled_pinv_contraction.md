# Scaled `pinv` Contraction Rejection - `frankenscipy-vgs9h`

## Lever

Replace `pseudo_inverse_from_svd`'s dense diagonal multiply:

```text
V * Sigma^+ * U^T
```

with a pre-scaled `V` matrix followed by one GEMM:

```text
(V * Sigma^+) * U^T
```

The lever preserves the same SVD factors and public `SVDFallback` certificate,
so it was only a post-SVD materialization candidate for `pinv`; it does not
address default `lstsq` or the full blocked-bidiagonal SVD target.

## Baseline

Existing current-head RCH criterion baseline on `ts1`:

- artifact: `baseline_pinv_512x256_criterion_rch.txt`
- command: `cargo bench -p fsci-linalg --bench linalg_bench --locked -- pinv/512x256 --noplot`
- `pinv/512x256`: `[85.492 ms 86.017 ms 86.559 ms]`

## Candidate Measurements

Scaled-contraction candidate captures:

- artifact: `baseline_pinv_512x256_confirm_rch.txt`
- worker: `ts1`
- `pinv/512x256`: `[292.81 ms 295.12 ms 297.26 ms]`

Tracked sample-size-10 capture:

- artifact: `baseline_current_pinv_512x256_criterion_rch.txt`
- worker: `ts2`
- `pinv/512x256`: `[453.51 ms 455.60 ms 457.56 ms]`

## Proof

The bit-identity proof passed for a focused small matrix:

- artifact: `cargo_test_pinv_contraction_bit_identity_rch.txt`
- result: `1 passed; 0 failed`

The full-rank rectangular golden payload is now available for future candidates:

- artifact: `golden_pinv_full_rank_rectangular_restored_payload.txt`
- SHA-256: `bb603e9c2452a8562c6f399ff2bce5a21b481e93080ff4ca9685e4c2e9bfe185`

## Decision

Reject. The candidate preserves the targeted output bits, but it regresses
`pinv/512x256` by roughly `3.4x` on the same `ts1` worker baseline
(`86.017 ms -> 295.12 ms`). The production source was restored to the existing
dense-diagonal multiply.

Score: `0.25 = Impact 1 * Confidence 1 / Effort 4`.

## Next Primitive

Continue `frankenscipy-vgs9h` with the real full-scope primitive: safe-Rust
blocked Householder bidiagonalization with compact reflectors, GEMM-backed
trailing updates, and deterministic bidiagonal SVD/vector reconstruction.
