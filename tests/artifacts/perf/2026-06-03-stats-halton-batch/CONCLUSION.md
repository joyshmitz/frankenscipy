# Halton 4D Batch Specialization Conclusion

Bead: `frankenscipy-rlcl4`

## Kept Lever

`HaltonSampler::sample` now dispatches only the default 4D prime set
`[2, 3, 5, 7]` to a fixed-base safe-Rust batch path. All other dimensions keep
the generic loop. The 4D path still emits row-major coordinates in base order
2, 3, 5, 7 and advances `next_index` once per emitted row with saturating
arithmetic.

## Performance

Focused RCH Criterion command:

`rch exec -- cargo bench -p fsci-stats --bench stats_bench --locked -- qmc_sampling/halton_4d/4096 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot`

- Before: `[238.55 us, 243.66 us, 248.22 us]`
- After: `[97.040 us, 99.507 us, 103.43 us]`
- Focused median speedup: `2.45x`

Post-change broad fsci-stats reprofile:

- `time_series/psd_welch/4096_w128_o64`: `518.65 us`
- `qmc_sampling/sobol_2d/4096`: `337.21 us`
- `qmc_discrepancy/mixture/512x2`: `284.49 us`
- `qmc_discrepancy/centered/512x2`: `198.45 us`
- `qmc_sampling/halton_4d/4096`: `90.627 us`

## Isomorphism Proof

- Ordering: row-major output preserved with coordinate bases 2, 3, 5, 7.
- Tie-breaking: unchanged; no comparisons or ordering decisions exist.
- Floating point: fixed-base helpers preserve the same digit recurrence as the
  generic `radical_inverse` loop.
- RNG/global state: unchanged; Halton sampling remains deterministic and does
  not touch RNG or global state.
- State transition: `next_index` uses one saturating increment per emitted row.
- Golden: before/after `halton_4d_4096` output is byte-identical.
- Golden sha256: `b8483708b0ed4f5becc8f7a11560adb063b7926c98fabf54fe8c95bfff4f2a44`.

## Validation

- RCH focused bit-regression: passed.
- RCH `cargo clippy -p fsci-stats --all-targets --locked -- -D warnings`: passed.
- `cargo fmt -p fsci-stats --check`: passed.
- `ubs crates/fsci-stats/src/qmc.rs crates/fsci-stats/src/bin/perf_stats.rs crates/fsci-stats/benches/stats_bench.rs`: passed.

## Score

`8.0 = impact 4 * confidence 4 / effort 2`

Verdict: kept; Score >= 2.0.
