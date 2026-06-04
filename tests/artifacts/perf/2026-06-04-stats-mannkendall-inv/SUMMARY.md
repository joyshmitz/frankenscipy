# mannkendall: O(n²) sign sum → O(n log n) merge-sort inversion count

Bench: `rank_correlation` (fsci-stats), rch ts2. Function: `mannkendall`.

## Lever
The Mann-Kendall S statistic, `S = #{i<j: x_j>x_i} - #{i<j: x_j<x_i}`, was an
O(n²) double sign loop. For non-NaN data it equals
`S = tot - tied_pairs - 2*inversions`, so it collapses to two O(n log n)
merge-sort passes reusing the primitives added for kendalltau
(`kendall_tie_pairs`, `kendall_strict_inversions`). Dispatched: fast identity
for n >= 256 NaN-free, the original O(n²) loop otherwise (NaN pairs contribute 0,
which the tie/inversion identity does not model; small n stays on the loop too).

## Isomorphism
S is an exact integer either way, so the downstream tau, variance, z, p-value,
and trend are bit-for-bit unchanged. Proven by
`mannkendall_inversion_s_matches_naive_loop` (fast S == naive S across
n ∈ {256,257,512,1000,2048} and tie densities continuous→heavy). clippy + fmt
clean.

## Benchmark (rch ts2, same-worker A/B: forced-naive vs fast)
| case                 | naive (O(n²)) | fast (O(n log n)) | Score |
|----------------------|---------------|-------------------|-------|
| mannkendall / 2048   | 1.578 ms      | 105.2 µs          | 15.0x |
| mannkendall / 4096   | 6.190 ms      | 299.5 µs          | 20.7x |

Second win from the inversion-count primitive (kendalltau was the first).
Remaining O(n²) rank stat: `somersd`. `theilslopes` is inherently O(n²)
(all-pairs slope median, matches scipy) — leave it.
