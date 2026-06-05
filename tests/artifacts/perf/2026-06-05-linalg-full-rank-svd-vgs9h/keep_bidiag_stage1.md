# Golub-Kahan Bidiagonal Reduction Stage 1 - `frankenscipy-s0r2q`

## Scope

This is the first private stage for `frankenscipy-vgs9h`, the profile-backed
full-rank rectangular SVD replacement target. It does not wire the new path into
public `svd`, `pinv`, or `lstsq`; public behavior and performance remain on the
existing fallback until the bidiagonal solver, vector reconstruction, and blocked
panel stages are proven.

## Lever

Add a safe-Rust unblocked Golub-Kahan Householder bidiagonal reduction for
`m >= n` matrices:

- returns diagonal and superdiagonal entries,
- retains left and right Householder reflectors,
- reconstructs the upper-bidiagonal matrix for proof tests,
- rejects wide matrices and non-finite inputs before factorization.

No unsafe code, C BLAS/LAPACK, Tokio, or external runtime dependency was added.

## Profile Context

The motivating public bottleneck remains the existing `nalgebra::SVD` route:

- `lstsq/512x256`: `[80.046 ms 80.745 ms 81.503 ms]` on `ts1`
- `pinv/512x256`: `[85.492 ms 86.017 ms 86.559 ms]` on `ts1`

This stage is an enabling primitive for the blocked-bidiagonal SVD route in
`blocked_bidiag_execution_contract.md`. Since the helper is private and not on a
public hot path, no speedup is claimed for this commit.

## Isomorphism

- Ordering/tie-breaking: no public singular values or vectors are emitted yet.
- Floating point: public API arithmetic is unchanged; the private proof checks
  reconstruction error and orthogonality only.
- RNG: none.
- Certificates/errors: public `SVDFallback` behavior is unchanged; private helper
  uses local rejection for unsupported wide or non-finite proof inputs.

## Golden Payload

Artifact: `golden_bidiag_golub_kahan_stage1_payload.txt`

SHA-256:

```text
c9aa3bec7c677f420ba88d2676e675423702e20ecf4239186c9059732e2c4ad4
```

The golden payload records shape, reflector counts, reconstruction error,
orthogonality errors, diagonal entries, and superdiagonal entries for a
deterministic `7x4` full-rank matrix.

## Validation

- RCH `cargo test -p fsci-linalg --lib bidiagonal_reduction --locked -- --nocapture`
  passed `2` focused tests.
- RCH `cargo test -p fsci-linalg --lib bidiag --locked -- --nocapture` passed `6`
  Stage 1 tests and emitted the golden payload.
- `cargo fmt -p fsci-linalg --check` passed.
- RCH `cargo check -p fsci-linalg --all-targets --locked` passed.
- RCH `cargo clippy -p fsci-linalg --lib --bins --tests --benches --no-deps --locked -- -D warnings`
  passed.
- `ubs crates/fsci-linalg/src/lib.rs` exited `0` with zero critical issues and
  the existing warning inventory.

## Decision

Keep Stage 1. The source change is not a standalone public performance win; it
is the proof harness and first safe-Rust primitive required before replacing the
full-rank rectangular SVD route. Next ready stages are deterministic bidiagonal
SVD (`frankenscipy-ffmf8`) and blocked Householder panels
(`frankenscipy-ox9ly`).
