# Direct Pinv Sigma Scaling Primitive Selection

- Bead: `frankenscipy-8l8r1.27`
- Profile target: `baseline_pinv/1000x500`
- Focused baseline: `316.20 ms`
- Golden SHA before edits: `271c9ee685150a31f31ca47867f9b2264eaa254542b1ea49907242bb895bc1cc`

## Selected Lever

Eliminate the dense diagonal `sigma_pinv` matrix and the first dense multiply in `pseudo_inverse_from_svd`. The current formula computes `V * Sigma * U^T` by materializing `Sigma` as a `p x p` matrix, multiplying `V * Sigma`, then multiplying by `U^T`. Because `Sigma` is diagonal, `V * Sigma` is equivalent to scaling each column of `V` by the corresponding reciprocal singular value (or zero/NaN under the existing threshold rules), followed by the same final dense multiply by `U^T`.

## Exclusions

- Not an SVD algorithm replacement.
- Not randomized or truncated SVD.
- Not QR/TSQR or a least-squares policy change.
- Not a threshold, tolerance, rank, or certificate policy change.
- Not validation, API, output layout, RNG, tie-breaking, or global-state behavior.
- No C BLAS, MKL, LAPACK, XLA, unsafe code, or fast-math contraction.

## Isomorphism Obligations

Validation and SVD invocation stay identical. Singular values are traversed in the same order, threshold decisions use the same predicate, and the final output matrix has the same row/column layout. For kept singular values, direct scaling computes the same `V[:, i] * (1 / s_i)` values that the diagonal matrix multiply would produce before the unchanged final multiply by `U^T`. For rejected singular values, the scaled column is `+0.0`, matching the effective dense diagonal contribution. For NaN singular values, the column is multiplied by `NaN`, preserving fail-closed propagation.

## Keep Gate

Keep only if the after benchmark shows a real `baseline_pinv/1000x500` median win versus `316.20 ms`, the stable golden SHA remains unchanged, RCH release pinv tests pass, and Score is at least `2.0`.

Target score: `7.5 = impact 5 * confidence 3 / effort 2`.
