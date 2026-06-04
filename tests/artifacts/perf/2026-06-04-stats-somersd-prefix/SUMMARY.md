# somersd: O((R·C)²) per-cell quadrant sums → O(R·C) 2D prefix sum

Bench: `somersd` (fsci-stats), rch ts2. Function: `somersd`.

## Lever
The concordant/discordant/variance accumulation called `somers_aij` and
`somers_dij` per table cell, each of which re-summed two table quadrants from
scratch — O((R·C)²) overall. For a rankings input over distinct values the table
is n×n, i.e. O(n⁴). Replaced with a single 2D prefix sum (O(R·C) to build) that
yields every quadrant total in O(1), collapsing the pass to O(R·C). Gated on the
table being exact non-negative integer counts with total < 2^53 (always true for
the rankings path); non-integer user tables keep the original exact summation.

## Isomorphism
For integer counts every quadrant total is an exact integer, so the prefix sums
equal the per-cell quadrant sums bit-for-bit, and the row-major accumulation
order of P/Q/a_term is preserved — statistic and p-value are unchanged. Proven by
`somers_prefix_pqa_matches_direct_quadrant_sums` (asserts P/Q/a_term `.to_bits()`
equality across shapes up to 100×100 and count magnitudes up to 1000). The 3
existing scipy-reference somersd tests pass; clippy + fmt clean.

## Benchmark (rch ts2, same-worker A/B: forced-naive vs prefix)
| case                | naive (O(n⁴)) | prefix (O(n²)) | Score |
|---------------------|---------------|----------------|-------|
| somersd rankings/64 | 4.548 ms      | 70.96 µs       | 64.1x |
| somersd rankings/128| 87.955 ms     | 269.8 µs       | 326x  |

Naive scales ×16 per doubling (O(n⁴)); prefix ×3.8 (O(n²)) — the ratio grows
without bound with table size. Biggest complexity-class win of the rank-stat
campaign (kendalltau 20-32x, mannkendall 15-21x were O(n²)→O(n log n)).
