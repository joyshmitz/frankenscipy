# Multi-RHS TRSM for the matrix inverse (inv_blocked)

## Lever
`inv_blocked` (gated at n>=1024) factored once with the blocked LU, then solved the n
identity-columns ONE AT A TIME via the scalar single-RHS `lu_subst_factored` — the dominant
O(n^3) phase of inversion was fully scalar (BLAS-2 per column). Replace it with a multi-RHS
triangular solve (`trsm_inv_columns`) that carries all RHS columns of a block together and
vectorizes the forward/back substitution 8-wide ACROSS the columns (`block_axpy_sub`,
`Simd<f64,8>`), still parallelized over column-blocks across threads. This is the BLAS-3 TRSM.

## Isomorphism (bit-identical)
Each element keeps the IDENTICAL incremental `s -= L[i][p]*B[p][j]` (forward) / `s -=
U[i][p]*B[p][j]` then `/U[i][i]` (back) over monotonic p that the per-column scalar path
performs — only the column dimension is SIMD-batched. Proof:
- `cargo test -p fsci-linalg --release --lib -- --include-ignored` = **430 passed, 0 failed**;
  `flat_lu_golden_digest` (which DIGESTS inv_blocked's output) still **0x2fc8ed294ef0427c**.
- probe checksum byte-identical before/after: **1.984582e-2**.

## Benchmark (perf_solve `inv` mode, n=2000, repeats=5)
Same-worker A/B (vmi1152480), `inv_blocked` only (stash lib.rs for before, keep the probe):
| version                     | per_call_ms |
|-----------------------------|-------------|
| before (scalar per-column)  | **5109.59** |
| after  (multi-RHS TRSM)     | **1875.16** |
=> **2.72x** (the before ran on a busier slot — true compute speedup ~2x+).
Cross-worker after points: 864 ms (vmi1227854), criterion baseline_inv/2000 2.52 s (vmi1264463).

The probe gained an `inv` mode (perf_solve.rs) for fast direct timing (criterion's
baseline_inv/2000 is ~244 s for 100 samples — too slow across the session cadence).
