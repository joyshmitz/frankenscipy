# Blocked Bidiagonal SVD Execution Contract - `frankenscipy-vgs9h`

## Measured Target

General full-rank rectangular SVD-family work remains on the `nalgebra::SVD`
path:

- `lstsq/512x256`: `[80.046 ms 80.745 ms 81.503 ms]` on `ts1`
- `pinv/512x256`: `[85.492 ms 86.017 ms 86.559 ms]` on `ts1`

The low-rank tall gates already handle reconstruction-certified low-rank inputs.
This contract targets the default/full-rank rectangular path only.

## Rejected Shallow Substitutes

- QR/TSQR default replacement: changes `lstsq` solution bits, singular values,
  rank thresholds, and certificate semantics unless full SVD stays on the hot
  path.
- Gram/eigen SVD surrogate: squares conditioning and was measured flat or
  regressive.
- Generalized low-rank gate: would publish false zero tails/rank for full-rank
  inputs.
- Post-SVD `pinv` materialization tweaks: exact-output preserving, but slower
  than the available baseline.
- Randomized/truncated SVD: cannot preserve full singular vector/value contract.

## Recommendation Card

Primitive: safe-Rust Golub-Kahan Householder bidiagonalization with blocked
panel/trailing updates, followed by deterministic bidiagonal SVD and vector
reconstruction.

Source lineage:

- Alien graveyard `9.6 Communication-Avoiding Algorithms`: panelized QR/LU
  structure and BLAS-3 trailing updates.
- Alien artifact numerical-linear-algebra family 34: SVD proof obligations,
  condition/rank certificate, orthogonality, and reconstruction checks.

Expected value:

- Impact: 5
- Confidence: 3
- Effort: 5
- Score: 3.0

Target ratio: at least `2x` on `lstsq/512x256` and `pinv/512x256` before
attempting the original `>=8x` tall profile row.

## Implementation Stages

1. Unblocked Golub-Kahan bidiagonal reduction for `m >= n`, returning `B`,
   left reflectors, and right reflectors. It is not wired to public APIs until
   factorization reconstruction and orthogonality proofs pass.
2. Deterministic bidiagonal singular-value/vector solver with strict ordering,
   sign canonicalization, convergence caps, and fallback to `nalgebra::SVD`.
3. Thin `U`, `S`, `Vt` reconstruction that exactly preserves public dimensions,
   rank thresholds, residual policy, `SVDFallback` certificate action, and error
   classes.
4. Blocked panel implementation: accumulate compact WY reflectors and apply
   trailing updates using existing safe-Rust GEMM/matmul kernels.
5. Public gate: wire only `m >= n`, finite, full-rank rectangular inputs whose
   reconstruction, orthogonality, and singular-value checks pass; otherwise
   fallback to the current `safe_svd`.

## Proof Obligations

- Ordering preserved: singular values sorted descending with deterministic tie
  handling.
- Tie-breaking: equal/near-equal singular values use the current fallback unless
  canonicalization matches the golden payload.
- Floating point: public output must satisfy the existing SciPy tolerance tests;
  full-rank golden payload SHA must remain stable for accepted gates.
- RNG: none.
- Certificates: public routes keep `SVDFallback` action and current rcond/rank
  semantics.
- Fallback: any non-finite input, convergence failure, rank ambiguity, or
  reconstruction/orthogonality breach returns to `safe_svd`.

## Verification Gates

- RCH `cargo test -p fsci-linalg --lib bidiag --locked -- --nocapture`.
- RCH `cargo test -p fsci-linalg --lib svd --locked -- --nocapture`.
- RCH `cargo test -p fsci-linalg --lib pinv --locked -- --nocapture`.
- RCH `cargo test -p fsci-linalg --lib lstsq --locked -- --nocapture`.
- Golden SHA for `pinv_full_rank_rectangular_golden_payload`.
- Criterion before/after for `lstsq/512x256` and `pinv/512x256` on the same
  worker when possible.
- UBS on `crates/fsci-linalg/src/lib.rs` and any touched linalg harness.

## Fallback Trigger

Reject and restore the public path if the first wired gate does not clear
Score `>= 2.0` on same-worker RCH, even if factorization tests pass.
