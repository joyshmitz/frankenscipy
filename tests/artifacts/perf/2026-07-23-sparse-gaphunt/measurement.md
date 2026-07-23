# Sparse CG/SpMV gap-hunt vs scipy (2026-07-23, thinkstation1, cc)

2D 5-point Laplacian, side 80/120 (n=6400/14400). fsci vs scipy.sparse 1-thread.

| fn | fsci | scipy | fsci vs scipy |
|---|---|---|---|
| spmv n=6400 | 29.3 µs | 24.7 µs | 1.19x slower |
| spmv n=14400 | 61.2 µs | 44.7 µs | 1.37x slower |
| cg n=6400 | 5.13 ms | 11.63 ms | 2.27x FASTER |
| cg n=14400 | 16.13 ms | 25.62 ms | 1.59x FASTER |

## FINDING
CG: fsci FASTER (pure Rust, no per-iteration Python matvec overhead; scipy.sparse.linalg.cg
is Python-wrapped). SpMV: fsci ~1.2-1.4x slower — a minor memory-bandwidth-bound kernel gap
(scipy's C SpMV is cache-tuned; fsci is close, ~19 GB/s). Not a clean lever (bandwidth-bound).

## COMPLETE COMPETITIVE MAP (all this session's gap-hunts)
| surface | fsci vs scipy | status |
|---|---|---|
| dense factorizations (cholesky/eigh/svd/qr/schur) | ~2-2.7x SLOWER | BLOCKED (vndri / WY 2o0vp / D&C) |
| special functions (array) | FASTER (1.8-13.7x) | fsci wins (parallel+no-Python) |
| sparse CG | FASTER (1.6-2.3x) | fsci wins (no-Python) |
| sparse SpMV | ~1.3x slower | minor bandwidth kernel |
| stats presort / matrix-fn adaptive / orthopoly | harvested (KEEPs) | done |

CONCLUSION: fsci's vs-scipy perf gaps are CONCENTRATED entirely in the dense LAPACK-backed
factorizations (all ~2x, one root cause = serial per-step Householder reduction, blocked on
vndri or the WY-blocked rewrite 2o0vp). Every OTHER measurable surface fsci is
competitive-to-winning. The next real perf win = the heavy dense structural work (2o0vp).
