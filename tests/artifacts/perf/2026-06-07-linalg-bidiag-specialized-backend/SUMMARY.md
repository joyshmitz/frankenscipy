# Keep: Private Tridiagonal QR Bidiagonal SVD Backend

Bead: `frankenscipy-8l8r1.47`

## Target

After `frankenscipy-qxuhx` parallelized left-reflector replay, the remaining
large-panel private bidiagonal SVD backend still spent real time in the dense
Gram `symmetric_eigen` route. The profile-backed target was the
`bidiag_svd_symmetric_eigen_route_perf_probe` path for the deterministic
`1024x512` bidiagonal SVD stage.

Fresh baseline on RCH `ts1`:

```text
jacobi_reference_ms=3548.469314
symmetric_eigen_route_ms=161.842119
reconstruction_error=2.32830643653869629e-10
u_column_orthogonality_error=9.71001057337161910e-13
vt_orthogonality_error=6.88338275267597055e-15
jacobi_reference_digest=0xd485296937b9f15f
svd_digest=0xaa1679aecf5d04b6
```

## Lever

For large bidiagonal Gram panels (`cols >= 128`), skip dense Gram
tridiagonalization and run a private symmetric tridiagonal QR eigensolver
directly from:

```text
gram_diagonal[i] = d[i]^2 + e[i-1]^2
gram_offdiagonal[i] = d[i] * e[i]
```

Small panels and QR nonconvergence still fall back to the existing dense
`symmetric_eigen` path. The downstream singular-value ordering, source-index
tie-break, vector sign canonicalization, rank thresholds, certificate behavior,
public errors, and RNG absence are unchanged.

## Proof

Focused helper proof passed on RCH:

```text
test tests::tridiagonal_qr_eigen_reconstructs_bidiag_gram ... ok
```

Large backend proof/perf probe passed on the same worker as the baseline:

```text
worker=ts1
jacobi_reference_ms=3532.096876
symmetric_eigen_route_ms=78.642349
reconstruction_error=2.32830643653869629e-10
u_column_orthogonality_error=3.42614825399323308e-13
vt_orthogonality_error=1.37667655053519411e-14
jacobi_reference_digest=0xd485296937b9f15f
svd_digest=0x7c2787acb98e625f
```

The private digest changes because the backend vector basis changes, but the
probe checks singular values against the Jacobi reference, reconstruction, and
orthogonality. Public behavior stayed unchanged:

```text
PUBLIC_SVD_LSTSQ_PINV_GOLDEN sha256
1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225
```

## Rebench

Same-worker RCH `ts1`:

```text
before symmetric_eigen_route_ms=161.842119
after  symmetric_eigen_route_ms=78.642349
speedup=2.058
```

The follow-up broader replay probe ran on `vmi1149989`, so it is not the
same-worker score gate. It showed the next visible work has shifted back toward
the full reduction/replay envelope:

```text
serial_ms=309.867839
parallel_ms=269.011067
speedup=1.151878
digest=0x46e400c14112e593
```

## Validation

Passed:

- `cargo fmt -p fsci-linalg --check`
- `git diff --check -- crates/fsci-linalg/src/lib.rs`
- RCH `cargo check -p fsci-linalg --all-targets --locked`
- RCH `cargo clippy -p fsci-linalg --all-targets --no-deps --locked -- -D warnings`
- `ubs crates/fsci-linalg/src/lib.rs`

UBS reported zero criticals and the existing broad linalg warning inventory.

## Decision

Keep.

Score: `5.1 = Impact 2.06 * Confidence 5 / Effort 2`.

Next primitive: reprofile current main, then attack the next largest linalg
stage with a structural algorithmic lever rather than another replay/indexing
micro-tweak.
