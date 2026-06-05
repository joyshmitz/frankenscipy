# fsci-linalg solve: nalgebra single-threaded LU -> in-house parallel blocked LU

## Target (profile-backed; follows the parallel-GEMM win 4b9a8869)

After matmul was parallelized, solve/inv became the dominant dense-linalg gap: they
go through nalgebra's single-threaded LU. Probe (64-core rch worker):

| op | n=1024 | n=2048 |
| --- | ---: | ---: |
| matmul (parallel) | 37.7 ms | 160 ms |
| solve (nalgebra LU) | 129.6 ms | 5065 ms |

solve at 2048 was 31x slower than matmul at the same n — single-threaded LU.

## Lever (one) — our own LAPACK-class kernel

`lu_solve_blocked`: right-looking BLOCKED LU with partial pivoting (panel width
NB=128). Each panel is factored unblocked (single-thread, O(n·NB^2)), then the
trailing submatrix update A22 -= L21·U12 — the O(n^3) bulk of the work — runs through
the multithreaded flat-workspace GEMM (matmul_flat_workspace) on all cores. Wired as a
fast path in `solve()` for the plain large case (n>=1024, Strict mode, untransposed,
assume_a None/General, finite); a singular pivot or any unmet precondition falls
through to the unchanged portfolio LU solver, so all diagnostics (rcond, hardened
checks, special assumptions, transposition) keep exact behavior.

## Parity (tolerance — our LU vs the reference LU)

Partial pivoting reproduces the LAPACK/SciPy factorization, so x matches the reference
to rounding. New test lu_solve_blocked_matches_reference: vs nalgebra solve_general
max|dx| < 1e-7 AND residual ||Ax-b||_inf < 1e-9, across n=16/130/200/270 (straddling
NB, with partial last panels and pivot-forcing off-diagonals). fsci-linalg 348 passed /
0 failed; evidence_p2c002 solve parity passes; lib clippy + fmt clean. (Conformance
cases are < 1024 so they use the unchanged portfolio path.)

## Rebench (perf_solve_probe, 64-core worker)

| n | before (nalgebra LU) | after (blocked LU) | speedup |
| ---: | ---: | ---: | ---: |
| 1024 | 129.6 ms | 89.7 ms | 1.44x |
| 2048 | 5065.7 ms | 436.0 ms | 11.6x |

Win grows steeply with n (bigger trailing GEMMs -> more parallelism). Score >> 2.0.
NEXT: inv (still nalgebra, 10.9 s @2048) — solve against identity columns via the
same blocked LU + batched parallel triangular solves; and a blocked Cholesky for SPD.
