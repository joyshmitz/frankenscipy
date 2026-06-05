# fsci-linalg inv: nalgebra single-threaded LU-inverse -> in-house parallel blocked LU

## Target (bead frankenscipy-qiq6z; follows the blocked-LU solve b0766fd9)

inv was the last single-threaded dense gap (nalgebra LU then solve against I). Probe
(64-core worker): inv n=2048 = 10932 ms — 68x slower than the now-parallel matmul.

## Lever (one)

Factor A ONCE with the parallel blocked LU (lu_factor_blocked, refactored out of the
solve path: returns the L\U factors + row permutation), then solve A X = I over the n
identity columns. Each column is an independent forward/back substitution against the
shared factors; columns are split across threads via std::thread::scope. The O(n^3)
factorisation trailing-update is already parallel (flat-workspace GEMM); the O(n^3)
multi-column substitution is now parallel too. Wired as an inv() fast path (n>=1024,
Strict, assume_a None/General, finite); singular/edge -> portfolio inverse unchanged.

## Parity (tolerance)

New test inv_blocked_matches_reference: vs nalgebra inv max|dX| < 1e-7 AND
||A·A^-1 - I||_inf < 1e-9 across n=16/130/270 (straddling NB=128, partial panels).
fsci-linalg 349 passed / 0 failed; lib clippy + fmt clean. Conformance cases (<1024)
use the unchanged portfolio path.

## Rebench (perf_solve_probe, 64-core worker)

| op | before (nalgebra) | after (blocked LU) | speedup |
| --- | ---: | ---: | ---: |
| inv n=1024 | 431 ms | 151.7 ms | 2.84x |
| inv n=2048 | 10932 ms | 799.2 ms | 13.7x |

inv at 2048: 10.9 s -> 0.8 s. Win grows with n. Score >> 2.0.
With matmul, solve, and inv now all multicore safe-Rust kernels, the remaining dense
gaps are lstsq/pinv (QR/SVD, nalgebra) and a blocked Cholesky for SPD — next targets.
