# frankenscipy-zq5xy closeout

## Target

- Crate: `fsci-signal`
- Profile-backed hotspot: `wavelets/cwt_ricker/2048x32`
- Source: `tests/artifacts/perf/2026-06-02-signal-profile/reprofile_after_remez_broad_rch.txt`
  listed `cwt_ricker/2048x32` at `5.61 ms` median — second-largest remaining
  signal row after the already-optimized `remez`.

## One lever

`cwt` computes `convolve(data, &wavelet, Same)` once per width. On the FFT path
the forward FFT of the zero-padded `data` depends only on `fft_len`, yet it was
recomputed for every scale. The lever caches `fft(data_padded)` keyed by
`fft_len` and reuses it across widths.

For the 2048×32 ricker bench, every scale yields `wavelet_len = 10·width`
(10…320), so `full_len = 2048 + wavelet_len − 1` (2057…2367) all round to
`fft_len = 4096`. All 32 scales share one cached data FFT, eliminating 31
redundant length-4096 forward transforms (and 31 padded-input allocations).

The FFT/direct dispatch (`na·nb > 1000`), the padded-input bytes, the
pointwise-multiply order, the inverse transform, and the `Same` slice are all
reproduced exactly; the non-FFT path still delegates to `convolve`.

## Result (same-binary A/B, worker vmi from ab_bench.txt)

Both cases measured in a single criterion invocation to remove cross-worker
noise (per prior FFT-campaign finding):

- `cwt_ricker/2048x32` (cached):   `8.4621 ms  9.3490 ms 10.578 ms`
- `cwt_ricker_baseline/2048x32`:   `11.267 ms 11.590 ms 11.945 ms`
- Median: `11.590 ms → 9.349 ms`, **1.24× faster (19.3% reduction)**
- Confidence intervals do not overlap: cached `[8.46, 10.58]` lies entirely
  below baseline `[11.27, 11.95]`.

(The temporary `cwt_ricker_baseline` bench replicating the pre-lever per-width
`convolve` loop was reverted after measurement.)

## Behavior proof

- Golden before sha256: `6d7e109d6aa72ed23d593e2a77f12b0b6f825c339b4ede0346a35450ed24d27b`
- Golden after  sha256: `6d7e109d6aa72ed23d593e2a77f12b0b6f825c339b4ede0346a35450ed24d27b`
- `cmp golden_before.txt golden_after.txt`: identical (`GOLDEN_IDENTICAL`)
- Golden harness = `perf_signal` bin, `cwt(deterministic_signal(2048), ricker, 1..=32)`
  (mirrors the bench), 32×2048 coefficients dumped at `{:.17e}`.
- Isomorphism:
  - Width traversal order unchanged.
  - FFT-vs-direct dispatch threshold (`na·nb > 1000`) unchanged.
  - `fft(data_padded)` is bit-identical for a given `fft_len` because the
    padded-input bytes are identical across widths; caching only removes
    recomputation, never changes the value.
  - Pointwise multiply, inverse FFT, real-part truncation, and `Same` slice
    `[(nb-1)/2 .. +na]` are unchanged.
  - No RNG; no tie-breaking.

## Validation

- `cargo fmt -p fsci-signal --check`: pass (`FMT_CLEAN`)
- `rch exec -- cargo clippy -p fsci-signal --lib --bin perf_signal --bench signal_bench --locked -- -D warnings`: pass
- `rch exec -- cargo test -p fsci-signal --lib --locked`: 497 passed, 0 failed
- `ubs crates/fsci-signal/src/bin/perf_signal.rs`: exit 0
- `ubs crates/fsci-signal/src/lib.rs`: nonzero due to the pre-existing broad
  signal inventory tracked in `frankenscipy-0cbi6` (unrelated to this change).

## Score

- Impact: 4.0 (19% on the #2 signal hotspot, non-overlapping CIs)
- Confidence: 4.5 (golden bit-identical, clear mechanism)
- Effort: 1.5
- Score: 12.0
