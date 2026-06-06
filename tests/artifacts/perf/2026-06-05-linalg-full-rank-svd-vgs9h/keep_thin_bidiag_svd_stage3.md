# Stage 3 Keep: Thin U/S/Vt Reconstruction

Bead: `frankenscipy-egf12`

## Target

Continue the profile-backed `fsci-linalg` full-rank rectangular SVD-family chain
from `frankenscipy-vgs9h` by reconstructing private thin `U/S/Vt` factors from:

- the Golub-Kahan left reflector product,
- the deterministic bidiagonal SVD factors, and
- the right reflector product.

Public `svd`, `svdvals`, `lstsq`, and `pinv` remain on the existing `safe_svd`
route in this stage.

## Baseline

The parent public bottleneck remains the full-rank rectangular SVD family.
Stage 3 reused the Stage 2 public guard as the relevant baseline because this
lever is private:

- Stage 2 `ts1` guard before this stage:
  - `lstsq/512x256`: `[87.672 ms 88.588 ms 89.291 ms]`
  - `pinv/512x256`: `[90.169 ms 90.633 ms 91.402 ms]`
- Stage 3 current public guard artifact:
  `baseline_thin_svd_stage3_lstsq_pinv_512x256_criterion_rch.txt`

The Stage 3 guard file contains worker-variance captures from RCH. It is used
only to prove the public route is still monitored, not to claim a public speedup.

## Lever

Added a private safe-Rust `DeterministicThinSvd` primitive that:

- multiplies the deterministic bidiagonal `U` factor by the left Householder
  product,
- multiplies the deterministic bidiagonal `Vt` factor by the right Householder
  product,
- jointly canonicalizes `U` and `Vt` signs without changing reconstruction,
- exposes private pseudo-inverse and least-squares helpers for proof tests, and
- leaves all public APIs unwired.

No unsafe code and no C BLAS/LAPACK linkage.

## Alien And Math Lineage

- Alien graveyard `Communication-Avoiding Algorithms`: this is the reconstruction
  stage of the blocked Householder bidiagonal SVD route.
- Alien artifact numerical linear algebra contract: reconstruction accuracy,
  orthogonality, deterministic sign policy, rank-threshold behavior, and public
  golden output are all pinned before public routing can use the primitive.

## Proof

RCH focused tests passed on `ts1`:

```text
cargo test -p fsci-linalg --lib thin_bidiag_svd --locked -- --nocapture
```

Result: `4 passed; 0 failed`.

Private thin-SVD golden payload:

- `golden_thin_bidiag_svd_stage3_payload.txt`
- SHA-256: `086c0c88cc52d431b9a497f7da60d64a25f2acde49ab0b387f6c03f44547fc73`

Golden values:

- Jacobi sweeps: `4`
- reconstruction error: `9.76996261670137756e-15`
- thin `U` column orthogonality error: `1.88737914186276612e-15`
- `Vt` orthogonality error: `1.11022302462515654e-15`
- singular values:
  - `1.42729956856374152e1`
  - `8.54591244670228178e0`
  - `7.42642247299642033e0`
  - `6.32220315986258630e0`

Public behavior proof:

- RCH `public_svd_lstsq_pinv_golden_payload` passed on `ts1`.
- Public golden SHA-256:
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`
- Public `svdvals` stayed equal to public `svd` singular values under the
  focused golden test.

## Isomorphism

- Ordering: public singular-value ordering still comes from `safe_svd`; private
  Stage 3 values inherit deterministic descending order from Stage 2.
- Tie-breaking: public tie behavior is unchanged; private signs are jointly
  canonicalized across `U` and `Vt`, preserving `U * S * Vt`.
- Floating point: public route output is pinned by the Stage 3 public golden SHA.
  Private `lstsq` and `pinv` helpers match public routes within `1e-8` on the
  deterministic proof matrix.
- RNG: none.
- Certificates and errors: public rank thresholds, fallback certificates, and
  error classes remain on the current `safe_svd` route.

## Post-Change Guard

RCH Criterion public-route guard:

- `after_thin_svd_stage3_lstsq_pinv_512x256_criterion_rch.txt`
- `vmi1227854` capture:
  - `lstsq/512x256`: `[114.59 ms 115.49 ms 116.39 ms]`
  - `pinv/512x256`: `[118.34 ms 119.65 ms 121.03 ms]`

No public speedup is claimed in this stage because the primitive is private and
unwired. The keep is for the proven reconstruction primitive required by the
blocked-panel and public-wiring stages.

## Validation

- RCH `cargo check -p fsci-linalg --all-targets --locked`: passed
- RCH `cargo clippy -p fsci-linalg --all-targets --locked -- -D warnings`:
  passed
- `cargo fmt -p fsci-linalg --check`: passed
- `ubs crates/fsci-linalg/src/lib.rs`: exit `0`, zero critical issues

## Risk Boundary

This primitive still depends on the Stage 2 Gram/Jacobi bidiagonal SVD helper.
It remains acceptable only while unwired. Public wiring must fallback for
ill-conditioned inputs, ambiguous rank thresholds, clustered ties, convergence
limit hits, or reconstruction/orthogonality proof breaches.

## Verdict

Keep as proof-stage primitive.

Score: `3.0 = Impact 5 * Confidence 3 / Effort 5`.

Next deeper primitive: blocked Householder panels with GEMM-backed trailing
updates, then guarded public SVD-family routing.
