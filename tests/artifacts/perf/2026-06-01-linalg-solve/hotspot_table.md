# Ranked Hotspot Table — dense linear solve (fsci-linalg)

**Run ID:** `2026-06-01-linalg-solve`  ·  **git:** `47b634d1` (harness/profile added on top)
**Scenario:** `solve(A, b)` for a deterministic 1000×1000 diagonally-dominant
(well-conditioned) system; CASP selects `DirectLU`. See `fingerprint.json`.
**Success metric:** p95 wall-clock per cold solve; secondary: warm per-call ms, peak RSS.
**Profiler note:** `perf`/`samply` ring-buffer mmap is unusable in this env
(`mmap failed`; `perf record` writes a file but resolves 0 samples). Attribution
below comes from (a) a deterministic 3-mode **stage decomposition** and (b) a
**gdb stack sampler** (`sudo gdb -batch -ex bt`, 110 samples over a 35 s warm run).
Both are load-robust for *relative* ranking; absolute ms carry the host caveats below.

## Baseline (hyperfine, 25 runs, cold single solve, n=1000)

| metric | value |
|--------|-------|
| p50 | 157.0 ms |
| p95 | 178.1 ms |
| p99 | 179.4 ms |
| min / max | 137.6 / 179.7 ms |
| mean ± σ | 156.2 ± 11.7 ms (CV 7.5%, within ≤10% envelope) |
| User / System | 109.6 ms / **45.3 ms (29%)** |
| peak RSS | 55.4 MB |
| minor page faults (cold) | 15,765 |

Warm per-call (repeats=20, median of 5): **solve 100.8 ms · lu_factor 84.3 ms · lu_solve 102.3 ms**
⇒ LU factorization ≈ 84 ms, triangular solve ≈ 18 ms, CASP-diagnostics overhead ≈ 0.

## Hotspots

| Rank | Location | Metric | Value | Category | Evidence |
|------|----------|--------|-------|----------|----------|
| 1 | nalgebra LU rank-1 update `gauss_step`→`array_axcpy` (nalgebra 0.34.2 `lu.rs:355`, `blas_uninit.rs:49`), reached via `fsci_linalg::condition_diagnostics_with_assumption` `lib.rs:1137` (`matrix.clone().lu()`) | cumulative | **~75 ms/solve (~75% warm)** | CPU | `pmp_samples.txt`: `gauss_step` in 82/110 stacks; self#0 = `add` 37%, **`f64::clone` 22%**, idx `unchecked_add` 13%, `array_axcpy` 8% |
| 2 | Redundant full-matrix copies in solve setup: `solve_with_portfolio_internal` `lib.rs:1569` `a.to_vec()` → `dmatrix_from_rows` → `lib.rs:1137` `matrix.clone()` | alloc/copy | **~3×8 MB = 24 MB/solve; ~6,200 minor faults/solve; 45 ms (29%) cold system time** | CPU/alloc | `time_v_cold.txt` 15,765 faults; self#0 `clone` 22% + `__brk` 4%; code read |
| 3 | `matrix.clone()` before `.lu()` (`lib.rs:1137`) kept only so `dispatch_solve_action` can run `compute_backward_error(&matrix,…)` | copy | ~1×8 MB/solve (subset of #2) | alloc | code read `lib.rs:1137` + `dispatch_solve_action` |
| 4 | Triangular solve `LU::solve(&rhs)` in `dispatch_solve_action` (DirectLU arm) | cumulative | ~12–18 ms/solve (~12%) | CPU | `pmp_samples.txt` 13/110; `lu_solve − lu_factor` = 18 ms |
| 5 | CASP condition diagnostics beyond the LU (structural scan, `fast_rcond_from_lu`) | overhead | ~0 measurable | CPU | `solve` 100.8 ms ≈ `lu_solve` 102.3 ms; rcond reuses cached LU |

Each ranked row cites an artifact in this directory (`pmp_samples.txt`,
`stage_decomposition.txt`, `time_v_cold.txt`, `baseline_hyperfine.json`) or a
`crates/fsci-linalg/src/lib.rs` line.

## Host caveats (fold into any absolute-number claim)

- **CPU governor = `powersave`** and turbo state unknown — P-state jitter present. Not tuned (would need explicit OK).
- **Host loadavg ≈ 63** (active agent swarm). The single-threaded solve still held a clean core (CV 7.5%), but absolute ms may drift run-to-run; *relative* attribution is stable.
- **perf/samply unavailable** (ring-buffer mmap denied); attribution via stage timing + gdb sampler instead.
- `perf_event_paranoid` was momentarily set 1→1 (already 1) and left at its original value; no persistent kernel change.
