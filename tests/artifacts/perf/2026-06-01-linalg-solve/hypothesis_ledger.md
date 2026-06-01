# Hypothesis Ledger — dense linear solve (fsci-linalg)

Run `2026-06-01-linalg-solve`. Each candidate explanation marked
`supports` / `rejects` with the evidence that settled it.

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| CASP condition diagnostics are an expensive overhead on top of the solve | **rejects** | warm `solve` 100.8 ms ≈ `lu_solve` 102.3 ms; `fast_rcond_from_lu` reuses the cached LU; structural scan is O(n²). CASP *dispatch* is essentially free. |
| The LU is factored twice (once for diagnostics, once for the solve) | **rejects** | `dispatch_solve_action` DirectLU arm reuses `lu_cache.take()` (`lib.rs`); timing `solve` ≈ `lu_factor` + 17 ms, not 2× `lu_factor`. |
| LU factorization is the dominant cost and runs through nalgebra's *generic scalar* path (no SIMD/blocking) | **supports** | `gauss_step` in 82/110 gdb stacks (~75%); self#0 distribution is 37% fp-`add` + **22% `f64::clone`** + 13% index `unchecked_add` → ~35% of LU self-time is generic-dispatch overhead, not arithmetic. nalgebra 0.34.2 built without a BLAS/`matrixmultiply` backend. |
| Matrix marshalling causes redundant copies / allocation churn | **supports** | code: `a.to_vec()` (`lib.rs:1569`) + `dmatrix_from_rows` + `matrix.clone()` (`lib.rs:1137`) = 3 full-matrix copies/solve; `time -v` shows 15,765 minor page faults cold and ~6,200/solve warm; 45 ms (29%) cold system time; self#0 `clone` 22%, `__brk` 4%. |
| `Vec<Vec<f64>>` pointer-chasing during conversion is the dominant cost | **rejects (as dominant)** | conversion frames appear in only ~5/110 stacks. The cost is copy *volume* + allocation/first-touch, not row-pointer chasing. |
| Triangular solve (forward/back substitution) is a major cost | **rejects** | ~12% of samples; `lu_solve − lu_factor` = 18 ms. Real but secondary to factorization. |
| `compute_backward_error` (extra O(n²) residual) is why the original matrix is cloned | **supports** | DirectLU arm calls `compute_backward_error(&matrix, &x, &rhs)`; the matrix is retained (hence the `matrix.clone()` before consuming `.lu()`). Making backward-error opt-in removes one 8 MB copy/solve. |

## Headline

For a well-conditioned dense solve, ~**75%** of time is the **LU factorization**,
and it runs through nalgebra's **non-vectorized generic BLAS** (≈⅓ of that is
`clone`/index overhead, not arithmetic). The second lever is **redundant
full-matrix copies** in the CASP setup (~3×/solve, ~6,200 page faults/solve,
29% of cold wall-time in the kernel). CASP dispatch/diagnostics themselves are
**not** a cost. Hand these to the optimizer as beads — measurement only here.
