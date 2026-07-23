# eigh vs scipy gap-hunt (2026-07-23, thinkstation1, cc)

Same-box, scipy 1.17.1/numpy 2.4.3 pinned 1-thread (OMP/OPENBLAS=1) vs fsci criterion.

| n | fsci eigh (full) | scipy eigh 1-thread | ratio | scipy eigvalsh |
|---|---|---|---|---|
| 256 | 8.36 ms | 6.35 ms | 1.32x slower | 2.10 ms |
| 512 | 99.58 ms | 36.53 ms | **2.73x slower** | 11.19 ms |

RATIO GROWS with n (1.32x -> 2.73x) => not a flat kernel-constant gap.

fsci path: n<512 -> nalgebra symmetric_eigen; n>=512 (PUBLIC_NATIVE_EIGH_MIN_DIM) ->
symmetric_eigh_native (Householder tridiagonalization + inverse-iteration / QR-fallback
eigensolve + eigenvector accumulation). scipy = LAPACK dsyevd (divide-and-conquer).

The growing ratio is the eigenvector-computation scaling: LAPACK D&C is O(n^3) with far
better constants for eigenvectors than QR-iteration / inverse-iteration. Closing this
needs a pure-Rust divide-and-conquer symmetric eigensolver (Cuppen's algorithm) — a
major multi-session effort, kernel-quality-wall class (cf. the cholesky n=1000 wall).
eigvalsh (eigenvalues-only) correctly skips eigenvectors (nalgebra symmetric_eigenvalues).
NOT a single-lever target; filed for a focused effort.
