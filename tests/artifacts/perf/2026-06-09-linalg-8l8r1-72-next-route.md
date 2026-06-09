# Post-GEMM Reprofile Route

Commit under test: `a54137165d8b13db9ddd63425e573eb738147610`.

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- --noplot
```

Worker: `ovh-a`.

The command was stopped with SIGTERM after the default linalg Criterion rows
completed and before the heavy `baseline_*` group finished. The default rows are
usable for routing the next child bead.

## Top Default Rows

| row | median |
| --- | ---: |
| `eigh_dense/512x512` | 95.001 ms |
| `matmul/1024x1024` | 51.309 ms |
| `matmul/768x768` | 23.855 ms |
| `lstsq/512x256` | 14.674 ms |
| `pinv/512x256` | 13.538 ms |
| `inv/256x256` | 10.722 ms |
| `matmul/512x512` | 9.9499 ms |
| `solve/256x256` | 2.5665 ms |

## Route

The next target is the dense symmetric eigensolver residual. Do not repeat
output-copy sorting, `eigvalsh` values-only routing, or the medium GEMM dispatch
gate. The next lever should attack the symmetric reduction/backend itself:
blocked symmetric band reduction, cache-blocked tridiagonalization, or a
two-stage full-to-band plus band-to-tridiagonal route.

Required proof gate:

- Same-worker RCH baseline before source edits.
- Eigenvalue ordering and total-cmp tie behavior preserved.
- Eigenvector sign/tie policy preserved.
- Reconstruction `A = V diag(w) V^T` within the existing tolerance contract.
- Public golden digest for eigenvalues/eigenvectors unchanged.
- No RNG, no unsafe, and no external BLAS/LAPACK/MKL/XLA linkage.
- Keep only with campaign Score >= 2.0.
