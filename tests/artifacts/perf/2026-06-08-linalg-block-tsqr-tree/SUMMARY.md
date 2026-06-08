# Block-TSQR Tree/Replay Closeout

Bead: `frankenscipy-8l8r1.56`

## Decision

Reject the true block-TSQR tree/replay route as a runtime/public candidate.

The implementation preserved behavior under focused proof, but failed the
same-process RCH performance gate:

- worker: `vmi1227854`
- `bidiag_candidate_ms=75.329034`
- `tsqr_candidate_ms=92.134037`
- `candidate_speedup=0.817603`
- `bidiag_reconstruction_error=4.20943280232677353e-11`
- `tsqr_reconstruction_error=3.21733750752173364e-11`

The public tall route was restored to the accepted bidiag candidate. TSQR helper
code is kept under `#[cfg(test)]` only, with an ignored A/B probe, so production
runtime behavior does not route through the rejected primitive.

## Baseline

Public route baseline on RCH worker `vmi1153651`:

- `reference_lstsq_ms=174.584885`
- `routed_lstsq_ms=131.225285`
- `lstsq_speedup=1.330421`
- `reference_pinv_ms=134.914952`
- `routed_pinv_ms=118.656896`
- `pinv_speedup=1.137017`
- `lstsq_max_abs_diff=1.07647224467655178e-12`
- `pinv_max_abs_diff=2.28428387316625958e-14`

Large reducer baseline on RCH worker `vmi1153651`:

- `shape=1024x512`
- `elapsed_ms=606.096750`
- `digest=0x90cdd3f8f71ed2c1`

Public golden SHA:

```text
1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225
```

## Proof

Focused TSQR proof:

- `block_tsqr_tall_svd_candidate_matches_safe_svd_reference`: passed on RCH.
- Deterministic repeat digest matched for fixed input.
- Singular values nonincreasing.
- Reconstruction, U orthogonality, and Vt orthogonality guards passed.

Public golden after route restoration:

- `public_svd_lstsq_pinv_golden_payload`: passed on RCH.
- Extracted payload SHA remained
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`.
- Payload diff against baseline was empty.

Restored public route probe on RCH worker `vmi1227854`:

- `reference_lstsq_ms=139.933797`
- `routed_lstsq_ms=52.422991`
- `lstsq_speedup=2.669321`
- `reference_pinv_ms=154.700236`
- `routed_pinv_ms=55.206684`
- `pinv_speedup=2.802201`
- `lstsq_rank=256`
- `pinv_rank=256`
- `lstsq_max_abs_diff=1.07647224467655178e-12`
- `pinv_max_abs_diff=2.28428387316625958e-14`

## Isomorphism

- Ordering: public runtime route restored to the accepted bidiag candidate; TSQR
  is test-only and cannot affect public singular ordering.
- Tie-breaking: public `public_bidiag_svd_accepts` rank-gap and clustered-spectrum
  guards are unchanged.
- Floating point: public golden payload is bit-identical by SHA; route restoration
  keeps the accepted bidiag arithmetic path.
- RNG: none.
- Safety: safe Rust only; no C BLAS/LAPACK/MKL/XLA; no `unsafe`.

## Validation

- `cargo fmt --check --manifest-path crates/fsci-linalg/Cargo.toml`: passed.
- `ubs crates/fsci-linalg/src/lib.rs`: zero critical issues; broad pre-existing
  linalg warnings remain.
- RCH `cargo check -p fsci-linalg --all-targets --locked`: passed.
- RCH `cargo clippy -p fsci-linalg --all-targets --no-deps --locked -- -D warnings`: passed.
- Broad dependency clippy without `--no-deps` failed before linalg on unrelated
  `fsci-fft` `manual_is_multiple_of`.

## Next Route

Created `frankenscipy-8l8r1.57`: blocked BLAS-3 Golub-Kahan bidiag core after
TSQR reject. The next route is explicitly not TSQR, QR-first, compact-WY
composition, single-step fusion, replay cleanup, or thread fanout. It targets a
safe-Rust blocked/GEMM-shaped bidiag primitive with reconstruction, rank,
sign/order, and public-golden proof obligations.
