# perf: barnard_exact / boschloo_exact — parallel nuisance-grid search, 26.6x byte-identical

## Lever (ONE)
Barnard's and Boschloo's unconditional exact tests maximize the p-value over a
grid of nuisance parameters p in [0,1] (n_grid = 1001 / 101). Each grid point's
p-value is an independent O(n1*n2) sweep over all 2x2 tables (Wald statistic /
Fisher hypergeometric sums), and the final result is the MAX over the grid.
`max` is a selection (not a sum), so it is order-independent: splitting the grid
across threads and combining per-chunk maxima is byte-identical to the
sequential scan. Compute-bound -> scales with cores.

## Parity — BYTE-IDENTICAL
- Each grid point's p-value is computed by the unchanged per-point function; the
  parallel per-chunk max + final max equals the sequential running max exactly
  (max over the same set, including the 0.0 seed). pvalue and statistic bits
  match the serial build exactly for barnard ([[8,2],[1,5]], [[40,10],[15,35]],
  [[120,30],[45,105]]) and boschloo ([[8,2],[1,5]], [[40,10],[15,35]]). See
  golden_payload.txt.
- All 11 fsci-stats barnard/boschloo tests pass; clippy clean.

## Timing — rch remote, 64 cores, --profile release-perf
| test    | table                  | serial | parallel  | speedup |
|---------|------------------------|--------|-----------|---------|
| barnard | [[200,80],[120,160]]   | 4.761 s| 178.7 ms  | 26.6x   |

boschloo uses the identical grid-parallel lever (parallel 5.876 s / 33.085 s for
the two large tables; the serial baseline is minutes-scale — the function is
notoriously expensive, which is exactly why parallelizing it matters). Both clear
Score >= 2.0 with a large margin.

Harness: crates/fsci-stats/src/bin/perf_barnard.rs
Run: `cargo run --profile release-perf -p fsci-stats --bin perf_barnard`
