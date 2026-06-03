# Primitive Selection: Direct Rectangular Lstsq SVD Solve

## Profile Target

The post-packed-B RCH reprofile ranked `baseline_lstsq/1000x500` as the next SVD-family hotspot after the rejected `pinv` diagonal-scaling lever. The focused RCH keep gate is `832.97 ms` median for `baseline_lstsq/1000x500`.

## Selected Primitive

Use the same full SVD but solve the least-squares vector directly:

```text
x = V * Sigma^-1 * U^T * b
```

The current rectangular path materializes `pinv(A) = V * Sigma^-1 * U^T` and then multiplies by `b`. The direct solve removes the dense `cols x rows` pseudoinverse materialization and replaces it with two deterministic matrix-vector passes over the existing SVD factors.

## Graveyard And Artifact Grounding

- `alien-graveyard`: Communication-Avoiding Algorithms (§9.6) target dense numerical-linear-algebra data movement and cache traffic.
- `alien-artifact-coding`: Numerical Linear Algebra family recommends method selection, condition monitoring, factorization accuracy, and explicit proof obligations for SVD/least-squares paths.

## Exclusions

- Not an SVD algorithm replacement.
- Not randomized/truncated SVD.
- Not QR/TSQR.
- Not a threshold, rank, residual, warning, or certificate policy change.
- Not a benchmark-harness edit.
- No unsafe code and no external BLAS/LAPACK/XLA backend.

## Score Target

Score target: `6.0 = impact 4 * confidence 3 / effort 2`. Reject if golden proof changes or focused RCH timing lacks a real `baseline_lstsq/1000x500` win against `832.97 ms`.
