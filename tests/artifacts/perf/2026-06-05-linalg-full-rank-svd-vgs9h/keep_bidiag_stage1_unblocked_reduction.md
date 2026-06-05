# Stage 1 Keep - Unblocked Golub-Kahan Bidiagonal Reduction

Bead: `frankenscipy-s0r2q`

Parent profile target: `frankenscipy-vgs9h`, full-rank rectangular SVD-family
residual.

## Baseline

The parent SVD-family residual remains profile-backed:

- `lstsq/512x256`: baseline `ts1` `[80.046 ms 80.745 ms 81.503 ms]`;
  current `ts2` confirmation `[128.27 ms 128.58 ms 128.93 ms]`.
- `pinv/512x256`: baseline `ts1` `[85.492 ms 86.017 ms 86.559 ms]`;
  restored/current `ts2` guard `[134.23 ms 134.74 ms 135.29 ms]`.

## Lever

Implemented a private safe-Rust unblocked Golub-Kahan Householder
bidiagonalization for `m >= n`, returning:

- diagonal and superdiagonal arrays,
- explicit left and right Householder reflector data,
- a materialized upper-bidiagonal proof matrix.

The primitive is intentionally not wired into public `svd`, `svdvals`, `lstsq`,
or `pinv` routes in this stage.

## Isomorphism Proof

- Public ordering/tie-breaking/floating-point/RNG behavior: unchanged, because no
  public route calls the new primitive.
- Public full-rank `pinv` golden payload before/after is byte-identical.
- Public golden payload SHA-256:
  `bb603e9c2452a8562c6f399ff2bce5a21b481e93080ff4ca9685e4c2e9bfe185`.
- New bidiagonal proof golden SHA-256:
  `c9aa3bec7c677f420ba88d2676e675423702e20ecf4239186c9059732e2c4ad4`.

## Proof Results

RCH focused proof:

- `cargo test -p fsci-linalg --lib bidiag --locked -- --nocapture`
- Worker: `ts2`
- Result: `6 passed; 0 failed`
- Reconstruction error: `7.10542735760100186e-15`
- `Q` orthogonality error: `1.10873533584960339e-15`
- `V` orthogonality error: `4.44089209850062616e-16`

RCH public golden proof:

- `cargo test -p fsci-linalg --lib pinv_full_rank_rectangular_golden_payload --locked -- --nocapture`
- Worker: `ts2`
- Result: `1 passed; 0 failed`

## Benchmark Guard

Post-stage Criterion guard on `ts2`:

- `lstsq/512x256`: `[129.91 ms 130.16 ms 130.53 ms]`
- `pinv/512x256`: `[135.14 ms 135.87 ms 136.50 ms]`

This is not claimed as a public speedup; the reducer is not wired yet. The guard
only confirms this proof-stage change did not route public work through a slower
candidate path.

## Validation

- `cargo check -p fsci-linalg --all-targets --locked`: pass via RCH `ts2`.
- `cargo clippy -p fsci-linalg --all-targets --locked -- -D warnings`: pass via
  RCH `ts2` after a dependency doc-lint fix in `fsci-fft`.
- `cargo fmt -p fsci-linalg -p fsci-fft --check`: pass.
- `ubs crates/fsci-linalg/src/lib.rs crates/fsci-fft/src/transforms.rs`: exit 0,
  no critical issues.

## Score

Score: `3.0 = Impact 5 * Confidence 3 / Effort 5`.

Rationale: this is the first required primitive for the blocked-bidiagonal SVD
chain. It is not a public performance keep until the later solver,
reconstruction, blocked-panel, and public-wiring beads clear same-worker
Criterion gates.
