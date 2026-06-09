# frankenscipy-8l8r1.69: square dense-Gram backend rejected

## Target

- Bead: `frankenscipy-8l8r1.69`
- Profile-backed residual: square public `svd()` core after prior right-replay and scaled-acceptance keeps.
- Baseline stage split from RCH `vmi1227854`: reduction `124.218 ms`, bidiagonal backend `68.970 ms`, U replay `65.103 ms`, V replay `269.760 ms`.
- Candidate: square-only dense `B^T B` Gram eigensolver for the bidiagonal backend, with the current tridiagonal-QR backend as the same-process reference.

## Same-Process A/B

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 rch exec -- cargo test -p fsci-linalg --release --lib --locked bidiag_svd_square_dense_gram_backend_perf_probe -- --ignored --nocapture --test-threads=1
```

Worker: `vmi1227854`

```text
shape=512x512
reduction_digest=0x931317ab18199fc4
tridiagonal_reference_ms=122.746682
dense_gram_ms=172.893944
speedup=0.709954
reference_sweeps=874
dense_sweeps=0
singular_value_max_diff=4.77484718430787325e-12
reconstruction_error=3.49245965480804443e-10
u_column_orthogonality_error=7.03437308402499184e-13
vt_orthogonality_error=1.03250741290139558e-14
reference_digest=0x1e9f77ab44771f01
dense_digest=0x7d4ea636b859eec6
```

Raw retry artifact: `ab_square_dense_gram_backend_rch_retry1.txt`.

```text
shape=512x512
reduction_digest=0x931317ab18199fc4
tridiagonal_reference_ms=56.307296
dense_gram_ms=98.689885
speedup=0.570548
reference_sweeps=874
dense_sweeps=0
singular_value_max_diff=4.77484718430787325e-12
reconstruction_error=3.49245965480804443e-10
u_column_orthogonality_error=7.03437308402499184e-13
vt_orthogonality_error=1.03250741290139558e-14
reference_digest=0x1e9f77ab44771f01
dense_digest=0x7d4ea636b859eec6
```

## Decision

Reject. The candidate preserved the existing tolerance-level backend proof, but it was slower than the current tridiagonal backend in the same process (`0.709954x`, reproduced at `0.570548x` in the raw retry artifact). No source changes are retained.

Next route: attack a fundamentally different backend primitive, not another Gram/eigen shortcut. The next candidate should be a direct bidiagonal divide-and-conquer or dqds-style backend with explicit reconstruction, orthogonality, ordering, and public golden SHA gates.
