# Cholesky n=1000 fsci-vs-scipy head-to-head — CORRECTED (Threadripper fleet)

Date: 2026-07-24 · Agent: CopperFalcon (cc) · Host: rch worker AMD Ryzen Threadripper PRO 5975WX (64 threads, 128 MiB L3) · scipy 1.17.1 / numpy 2.4.3 · fsci release `+avx2,+fma`, production blocked path (`perf_chol_gate`, FORCE_BLOCKED)

## MEASUREMENT GOTCHA (caught a wrong conclusion)

Timing `sla.cho_factor(A.copy(), ...)` with the `A.copy()` INSIDE the timed region adds an
8 MB numpy copy (~8 ms at n=1000) to every sample — making scipy look **11.9 ms** when the
factorization itself is **3.26 ms**. That inflated reading briefly showed "fsci 1.35x FASTER",
which is FALSE. Always pre-make the input copies and time only `cho_factor(c, overwrite_a=True,
check_finite=False)`. (fsci's `perf_chol_gate` already times only the factor — `a` is built once
outside the loop — so the comparison must give scipy the same treatment.)

## Corrected numbers (fair: overwrite_a=True, copies pre-made, median of 20)

| n | fsci (GF/s) | scipy-default all-64-cores (GF/s) | fsci/scipy |
|---|---|---|---|
| 1000 | 8.79–9.13 ms (~37 GF/s) | **3.26 ms (102 GF/s)** | **2.73x SLOWER** |
| 2048 | ~53 ms (~52 GF/s, banked) | 55.06 ms (52 GF/s) | ~1.04x FASTER (crossover holds) |

Cholesky flops = n³/3. Both arms use their default threading (fsci work-gated internal;
scipy OpenBLAS all 64 cores). Thread-parity rule satisfied (counts + GF/s reported per arm).

## Interpretation — the gap is THREADING SCALING, not the kernel

fsci's per-core kernel is competitive (≈ scipy 1-thread); the 2.73x at n=1000 is that scipy's
OpenBLAS dpotrf SCALES to 102 GF/s across 64 cores on the small problem while fsci's
barrier-per-panel structure (factor → TRSM → SYRK, a `thread::scope` join at each panel) only
reaches 37 GF/s. The mandate's "~8x" is stale (real: 2.73x at n=1000, and it was a
default-vs-1-thread conflation historically); at n≥2048 fsci is at parity/faster.

- The SYRK is memory-bandwidth-bound → more parallelism can't lift it (vndri-rayon REJECT confirmed this).
- Closing 37→102 GF/s needs OVERLAPPING panel/TRSM/SYRK across cores with NO per-panel barrier —
  a **task-DAG tile scheduler** (PLASMA/MKL style), a multi-session restructure. A cheaper pool
  (vndri) does not change the barrier structure, which is why it was IN-FLOOR.

CONCLUSION: dense n=1000 is a confirmed ledgered blocker — the residual 2.73x is threading-scaling,
addressable only by the task-DAG restructure (multi-session) or accepted (fsci wins at n≥2048).
