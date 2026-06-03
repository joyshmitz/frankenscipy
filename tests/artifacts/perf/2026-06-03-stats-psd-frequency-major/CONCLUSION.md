# PSD Welch Frequency-Major Traversal Conclusion

Bead: `frankenscipy-kko0m`

## Profile-Backed Target

The post-Halton broad fsci-stats reprofile put
`time_series/psd_welch/4096_w128_o64` at the top remaining stats rows:

- Broad median: `518.65 us`
- Focused RCH baseline for this bead: `[717.61 us, 746.46 us, 799.07 us]`

## One Lever Tested

The candidate pre-windowed all overlapping Welch segments into a flat row-major
matrix, then traversed frequency -> segment -> sample so each cached twiddle row
could be reused across all segments.

This was separate from the earlier rejected segment-buffer reuse lever.

## Isomorphism Proof

- Output ordering stayed frequency-index ascending.
- For each frequency, segment powers were added in ascending segment order.
- For each segment-frequency DFT, sample order stayed ascending.
- Each DFT step still used the same cached twiddle values and `re += s * cos`,
  `im -= s * sin` recurrence.
- Validation, overlap clamping, RNG, tie-breaking, global state, and error
  behavior were unchanged.

## Golden Proof

RCH `perf_stats golden` before/after output was byte-identical.

- Before sha256: `85048a3c06ab045815cbeb238fee9e1e07a05c27ceed3c3782ec0fd5ea97c6b1`
- After sha256: `85048a3c06ab045815cbeb238fee9e1e07a05c27ceed3c3782ec0fd5ea97c6b1`
- Comparison: `golden_cmp=identical`

## Benchmark Gate

Focused RCH Criterion command:

`rch exec -- cargo bench -p fsci-stats --bench stats_bench --locked -- time_series/psd_welch/4096_w128_o64 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot`

- Before: `[717.61 us, 746.46 us, 799.07 us]`
- After: `[752.03 us, 946.90 us, 1.3931 ms]`
- Result: `0.79x` by medians; regression.

## Verdict

Abandoned. Production `psd_welch` was restored to the pre-lever source, leaving
`git diff -- crates/fsci-stats/src/lib.rs` empty.

Score: `0.0` because impact was negative.
