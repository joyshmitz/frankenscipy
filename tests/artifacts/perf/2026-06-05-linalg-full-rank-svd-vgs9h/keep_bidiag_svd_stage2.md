# Deterministic Bidiagonal SVD Stage 2 - `frankenscipy-ffmf8`

## Scope

This is Stage 2 for the profile-backed `frankenscipy-vgs9h` blocked-bidiagonal
SVD route. It adds a private deterministic singular solver for the upper
bidiagonal factors produced by the Golub-Kahan reduction. It is intentionally
not wired into public `svd`, `svdvals`, `pinv`, or `lstsq`; those routes still
use the existing `safe_svd` fallback until reconstruction, blocked panels, and
public proof gates are complete.

## Profile Context

The parent public bottleneck remains the full-rank rectangular SVD family:

- Original `ts1` `lstsq/512x256`: `[80.046 ms 80.745 ms 81.503 ms]`
- Original `ts1` `pinv/512x256`: `[85.492 ms 86.017 ms 86.559 ms]`

Current Stage 2 public guard, before accepting this private solver:

- `lstsq/512x256`: `[85.447 ms 86.395 ms 87.297 ms]`
- `pinv/512x256`: `[88.135 ms 88.661 ms 89.145 ms]`

Post-stage guard on the same worker stayed in the same public route band:

- `lstsq/512x256`: `[87.672 ms 88.588 ms 89.291 ms]`
- `pinv/512x256`: `[90.169 ms 90.633 ms 91.402 ms]`

No public speedup is claimed for this stage because the new solver is private.

## Lever

Add a private deterministic bidiagonal SVD helper:

- forms the symmetric tridiagonal Gram matrix for the upper bidiagonal factor,
- diagonalizes it with a local symmetric Jacobi solver,
- sorts singular values descending with deterministic index tie-breaks,
- canonicalizes vector signs,
- reconstructs compact left singular vectors for nonzero singular values,
- fills zero-singular-value left vectors deterministically by Gram-Schmidt.

This is a proof-stage bridge toward the communication-avoiding blocked
bidiagonal route. It is not the final public solver: public wiring must still
gate ill-conditioned or clustered cases back to `safe_svd`.

## Alien / Math Lineage

- Alien graveyard `9.6 Communication-Avoiding Algorithms`: the parent route is
  blocked Householder bidiagonalization with explicit numerical stability and
  convergence certificates.
- Alien artifact family 34, Numerical Linear Algebra: requires factorization
  accuracy, orthogonality checks, condition/fallback notes, and independent
  golden artifacts.

## Isomorphism

- Ordering preserved: public outputs are unchanged because no public route calls
  this helper. Private singular values are sorted descending with deterministic
  index tie-breaks.
- Tie-breaking unchanged: public tie behavior remains on `safe_svd`; private
  helper uses fixed index order for equal eigenvalues.
- Floating point: public `pinv` golden output is byte-identical to Stage 1.
- RNG: none.
- Certificates/errors: public `SVDFallback` certificate semantics are unchanged.

## Golden Payloads

Private bidiagonal SVD payload:

```text
b5515c3aeed29cb28cd478db936b6e0bd62cd90ef4edf06cc0aaa207345c1a1c
```

Public full-rank `pinv` payload:

```text
bb603e9c2452a8562c6f399ff2bce5a21b481e93080ff4ca9685e4c2e9bfe185
```

`cmp` between the Stage 1 and Stage 2 public `pinv` payloads returned `0`.

## Validation

- RCH `cargo test -p fsci-linalg --lib bidiag_svd --locked -- --nocapture`
  passed `5` focused tests.
- RCH `cargo test -p fsci-linalg --lib pinv_full_rank_rectangular_golden_payload --locked -- --nocapture`
  passed and reproduced the Stage 1 public SHA.
- `cargo fmt -p fsci-linalg --check` passed.
- RCH `cargo check -p fsci-linalg --all-targets --locked` passed.
- RCH `cargo clippy -p fsci-linalg --all-targets --locked -- -D warnings`
  passed.
- `ubs crates/fsci-linalg/src/lib.rs` exited `0` with zero critical issues and
  the existing warning inventory.

## Risk Boundary

The private helper obtains singular values through a symmetric Gram matrix of
the already-bidiagonal factor. That is acceptable only while unwired and covered
by reconstruction/orthogonality/golden tests. Public wiring must retain fallback
guards for ill-conditioned, ambiguous-rank, clustered-tie, convergence, or proof
breach cases. The deeper target remains a robust bidiagonal QR/divide-and-conquer
solver or a guarded proof that this helper is used only inside a certified safe
region.

## Score

Score: `3.0 = Impact 5 * Confidence 3 / Effort 5`.

This clears the keep threshold as an enabling primitive for the blocked
bidiagonal route. The next bead is `frankenscipy-egf12`, reconstruction of public
thin `U/S/Vt` from Golub-Kahan reflectors plus these bidiagonal factors.
