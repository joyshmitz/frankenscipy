# Sobol 2D Cached Direction Numbers

Bead: `frankenscipy-seo6h`

## Verdict

Shipped. The cached Sobol direction-number table is a single pure safe-Rust
optimization lever and cleared the Score threshold.

## Benchmark

Focused RCH Criterion target:

`cargo bench -p fsci-stats --bench stats_bench --locked -- qmc_sampling/sobol_2d/4096 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot`

Baseline on `vmi1264463`: `[648.26 us, 914.73 us, 1.3581 ms]`.

After on `vmi1227854`: `[307.27 us, 319.23 us, 325.81 us]`.

Median speedup: `2.86x`.

Broad post-change stats reprofile on `vmi1293453` measured:

- `time_series/psd_welch/4096_w128_o64`: `449.66us` median.
- `qmc_sampling/sobol_2d/4096`: `323.21us` median.
- `qmc_discrepancy/mixture/512x2`: `322.03us` median.

## Behavior Proof

Golden command:

`cargo run -p fsci-stats --bin perf_stats --locked --quiet -- sobol2-golden`

Golden output before and after was byte-identical.

SHA256:

`ff622ca915b745828cff4da1fa1954c628deb6eb76d49612a606a2034862a815`

Isomorphism contract:

- Gray-code bits are still consumed least-significant to most-significant.
- Direction words are still xored into the accumulator in the same order.
- Table values are generated from the old recurrence.
- Digital shift application remains after `sobol_bits`.
- Row-major output order, dimension order, and `next_index` advancement are unchanged.
- `bits_to_unit` remains the only floating-point conversion.
- No RNG, tie-breaking, global state, validation, or error behavior changed.

## Validation

- RCH focused Sobol tests: `6 passed`.
- RCH `cargo check -p fsci-stats --all-targets --locked`: passed.
- RCH `cargo clippy -p fsci-stats --all-targets --locked -- -D warnings`: passed.
- `cargo fmt -p fsci-stats --check`: passed.
- UBS on changed stats files: exit 0.

## Score

`8.0 = impact 4 * confidence 4 / effort 2`.
