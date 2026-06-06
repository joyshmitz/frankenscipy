# Keep: Symmetric-Eigen Large Bidiagonal SVD Route Stage 4k

Bead: `frankenscipy-z65tz`
Date: 2026-06-06

## Profile-backed target

Clean-source RCH reprofile after the rejected Stage 4j trials kept the private
`1024x512` deterministic bidiagonal SVD backend as the active bottleneck.

RCH `ts1` baseline:

```text
cargo test -p fsci-linalg --release --lib thin_bidiag_factor_replay_perf_probe --locked -- --ignored --nocapture
test wall time: 4.49s
reduction_digest=0x90cdd3f8f71ed2c1
reference_ms=596.870042
replay_ms=316.441932
speedup=1.886191
u_max_abs_diff=5.10702591327572009e-15
vt_max_abs_diff=2.33146835171282873e-15
reference_digest=0x6e44a30879443520
replay_digest=0x412adbf362e4362b
```

The unprinted backend cost dominated the wall time after reducer and factor
assembly work had already been optimized.

## Lever

Route private large `deterministic_bidiagonal_svd` panels (`cols >= 32`) through
a symmetric eigensolver over the existing bidiagonal Gram construction instead
of the bounded dense cyclic Jacobi backend.

The small-shape path stays on Jacobi, preserving exact private golden payloads
that exercise diagonal/rank-boundary behavior. Public `svd`, `svdvals`,
`lstsq`, and `pinv` remain on the existing `safe_svd` route.

## Same-worker benchmark

RCH `ts1`, same binary, same process:

```text
cargo test -p fsci-linalg --release --lib bidiag_svd_symmetric_eigen_route_perf_probe --locked -- --ignored --nocapture
shape=1024x512
reduction_digest=0x90cdd3f8f71ed2c1
jacobi_reference_ms=2940.723215
symmetric_eigen_route_ms=108.267028
speedup=27.161762
reconstruction_error=2.32830643653869629e-10
u_column_orthogonality_error=9.71001057337161910e-13
vt_orthogonality_error=6.88338275267597055e-15
jacobi_reference_digest=0xd485296937b9f15f
svd_digest=0xaa1679aecf5d04b6
```

The same-worker score is well above the keep threshold.

## End-to-end reprofile

RCH `ts1` after the route:

```text
cargo test -p fsci-linalg --release --lib thin_bidiag_factor_replay_perf_probe --locked -- --ignored --nocapture
test wall time: 1.03s
reduction_digest=0x90cdd3f8f71ed2c1
reference_ms=470.055207
replay_ms=241.299199
speedup=1.948018
u_max_abs_diff=4.88498130835068878e-15
vt_max_abs_diff=2.66453525910037570e-15
reference_digest=0x22223a463752097f
replay_digest=0x8f521a39638fb520
```

The bottleneck has shifted back toward reducer/factor replay and future public
wiring gates.

## Behavior proof

RCH focused proof:

```text
cargo test -p fsci-linalg --release --lib bidiag_svd_symmetric_eigen_route_reconstructs_medium_panel --locked -- --nocapture
result: passed
```

RCH private SVD family:

```text
cargo test -p fsci-linalg --release --lib bidiag_svd --locked -- --nocapture
result: 10 passed; 0 failed; 1 ignored
```

RCH public golden:

```text
cargo test -p fsci-linalg --release --lib public_svd_lstsq_pinv_golden_payload --locked -- --nocapture
result: passed
```

Public payload SHA-256 from the exact payload block remains:

```text
1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225
```

## Isomorphism

- Ordering: singular values remain sorted descending by value.
- Tie-breaking: equal-value order still uses stable source-index fallback.
- Signs: singular vectors still pass through deterministic sign
  canonicalization.
- Floating point: public routes are unchanged; private large-panel vector
  factors change digest because the internal backend changes, but the route is
  still private and reconstruction/orthogonality certificates pass.
- RNG: none.
- Public behavior: `svd`, `svdvals`, `lstsq`, and `pinv` golden payload SHA is
  unchanged.

## Validation

```text
cargo fmt -p fsci-linalg --check
git diff --check -- crates/fsci-linalg/src/lib.rs
ubs crates/fsci-linalg/src/lib.rs
RCH cargo check -p fsci-linalg --lib --locked
RCH cargo clippy -p fsci-linalg --all-targets --no-deps --locked -- -D warnings
```

Results:

- formatting and diff checks passed
- UBS: zero critical issues
- RCH `cargo check`: passed
- RCH clippy: passed

## Decision

Keep.

Score: `9.0 = Impact 5.0 * Confidence 4.5 / Effort 2.5`.

Next primitive after reprofile: public wiring guard for full-rank rectangular
routes, plus reducer/factor replay follow-up if public wiring exposes the
private path's remaining reducer cost.
