# Sobol 2D Incremental Gray-Code Sampling

Bead: `frankenscipy-8l8r1.7`

## Lever

Added a private `SobolSampler::sample_2d` fast path that computes the starting
raw Sobol bits once, emits row-major 2D samples, and advances raw bits with the
gray-code transition direction at `trailing_zeros(next_index_after_increment)`.
The generic non-2D path remains unchanged.

## Performance

Focused RCH Criterion target: `qmc_sampling/sobol_2d/4096`

| Run | Worker | Interval | Median |
| --- | --- | --- | --- |
| Baseline | `vmi1153651` | `[471.70 us, 496.97 us]` | `486.30 us` |
| After | `vmi1156319` | `[18.931 us, 20.532 us]` | `19.634 us` |

Focused speedup: `24.77x` by median.

Broad post-change RCH reprofile on `vmi1227854`:

| Row | Median |
| --- | --- |
| `time_series/psd_welch/4096_w128_o64` | `464.40 us` |
| `qmc_discrepancy/mixture/512x2` | `276.17 us` |
| `qmc_discrepancy/l2_star/512x2` | `232.68 us` |
| `qmc_discrepancy/centered/512x2` | `220.32 us` |
| `qmc_sampling/sobol_2d/4096` | `10.887 us` |

The bottleneck shifted away from Sobol 2D sampling; the next stats hotspot is
PSD Welch.

## Isomorphism Proof

- Ordering preserved: rows are emitted from current `next_index`, then advanced
  once per requested row.
- Dimension order unchanged: every row remains dimension 0 followed by
  dimension 1.
- Gray-code transition unchanged: row `i + 1` raw bits equal row `i` raw bits
  xor direction `trailing_zeros(i + 1)`.
- Digital shift unchanged: the same per-dimension shift words are xor'ed before
  conversion.
- Floating-point unchanged: `bits_to_unit` remains the only conversion path.
- RNG unchanged: no new RNG calls or seed changes.
- Tie-breaking unchanged: no comparisons were introduced.
- Saturation unchanged: when `next_index` reaches `u64::MAX`, subsequent rows
  keep emitting the saturated index just like the prior `saturating_add` loop.
- Golden output: before and after `sobol_2d_4096` payloads are byte-identical;
  SHA stayed `ff622ca915b745828cff4da1fa1954c628deb6eb76d49612a606a2034862a815`.

## Validation

- RCH `cargo test -p fsci-stats --lib --locked -- sobol --nocapture`: passed,
  7 tests.
- RCH `cargo check -p fsci-stats --all-targets --locked`: passed.
- RCH `cargo clippy -p fsci-stats --all-targets --locked -- -D warnings`:
  passed.
- `cargo fmt -p fsci-stats --check`: passed.
- `ubs crates/fsci-stats/src/qmc.rs crates/fsci-stats/src/bin/perf_stats.rs crates/fsci-stats/benches/stats_bench.rs`:
  exit 0, critical 0.

## Score

Final score: `10.0 = impact 5 * confidence 4 / effort 2`.

Verdict: shipped. Score is above the `2.0` keep threshold.
