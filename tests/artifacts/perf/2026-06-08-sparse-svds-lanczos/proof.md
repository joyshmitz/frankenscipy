# perf: svds power iteration → Lanczos/Arnoldi on AᵀA, 2.9–4.5x

## Lever (ONE)
`svds` (top-k singular triplets of a sparse matrix) used
**power-iteration-with-deflation on AᵀA**: one singular value at a time, up to
`max_iter` operator applications each. The top-k singular values of A are the
square roots of the top-k eigenvalues of the n×n SPSD matrix AᵀA, with right
singular vectors = its eigenvectors.

Reuse the shared `krylov_arnoldi_eigs` (the Lanczos/Arnoldi solver introduced for
`eigsh`), now generalized to take an **operator closure** instead of a matrix, and
drive it with `v ↦ Aᵀ(A v)` (`csc_matvec(a_csc, csr_matvec(a, v))`, using the
cached-CSC parallel transpose). This gets all k eigenpairs of AᵀA in O(ncv)
operator applications (ncv = max(2k+1, 20)) versus the previous O(k·max_iter).
σ_i = √(max(λ_i, 0)); u_i = A v_i / σ_i.

`eigs` and `eigsh` were updated to pass their matvec as a closure too — same
behavior, shared code path.

## Parity — tolerance (both converge to the same true singular spectrum)
- On a rectangular matrix with planted well-separated singular values
  (100, 88, 76, …), Lanczos and the old power iteration return **identical**
  top-k singular values (100.000160, 87.999995, 76.000048, …); both have
  machine-level residuals ‖A v − σ u‖ (~1e-18). See `golden_payload.txt`.
- All 310 `fsci-sparse` lib + 56 bin/doc tests pass, including the svds, eigsh,
  and eigs suites (svds_diagonal_known_singular_values, svds_identity,
  svds_rectangular, …).

## Timing — rch remote, 64 cores, `--profile release-perf`
`svds` on rectangular sparse, planted separated spectrum:

| m × n         | k | power iteration | Lanczos | speedup |
|---------------|---|-----------------|---------|---------|
| 2200 × 2000   | 6 | 5.287 ms        | 1.543 ms| 3.43x   |
| 8200 × 8000   | 6 | 18.265 ms       | 6.278 ms| 2.91x   |
| 20200 × 20000 | 8 | 66.472 ms       | 14.751 ms| 4.51x  |

Score ≥ 2.0 cleared. (Lower than eigsh's ~8.6x because each svds operator
application is two matvecs, A·v then Aᵀ·w — both already parallel via the
cached-CSC transpose + parallel `csr_matvec`.)

Harness: `crates/fsci-sparse/src/bin/perf_svds.rs`
Run: `cargo run --profile release-perf -p fsci-sparse --bin perf_svds`

## Notes
- Completes the sparse-eigensolver Krylov upgrade: eigs (already Arnoldi), eigsh
  (commit 52a85cc9), and now svds all share `krylov_arnoldi_eigs`.
- Pathologically-clustered spectra would need implicit restarts (ARPACK-style);
  single-shot Lanczos here, same caveat as eigsh.
