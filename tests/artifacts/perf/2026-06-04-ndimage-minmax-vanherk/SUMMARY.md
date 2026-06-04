# minimum/maximum_filter: full-footprint sort → separable sliding-window deque

Bench: `minmax_filter` (fsci-ndimage), rch ts2. Functions: `minimum_filter`,
`maximum_filter` (via `*_with_origins`).

## Lever
The N-D min/max filter gathered the full `size^ndim` footprint per output pixel
and `sort_by(total_cmp)`-ed it to pick rank 0 / last — O(N · size^ndim · log).
A min/max over a rectangular footprint equals the per-axis sequential min/max, so
it is replaced with `ndim` sliding-window passes, each a monotonic deque
(O(1) amortized per output, **independent of kernel size**) → O(N · ndim).

## Isomorphism
Comparisons use `total_cmp` — the same total order the rank filter sorts by — and
every neighbourhood value comes from the same `get_boundary`, so the separable
result is bit-for-bit identical to the full-footprint rank filter, including NaN
and signed-zero handling. Origin validation is against the kernel footprint
`[size; ndim]`, matching the rank filter (accept/reject parity).
- `separable_minmax_matches_rank_filter_byte_for_byte`: asserts `.to_bits()`
  equality vs `rank_filter_index_with_origins` over 1D/2D/3D shapes, all 4
  boundary modes, even/odd sizes, origins (incl. extremes), min and max, on
  inputs containing NaN and ±0.0.
- 65 filter + 12 grey-morphology + 12 rank tests pass (grey erosion/dilation are
  built on min/max). clippy + fmt clean.

## Benchmark (256x256 image, rch ts2, same-worker A/B: naive rank vs separable)
| case            | naive (sort)  | separable deque | Score |
|-----------------|---------------|-----------------|-------|
| minimum size=7  | 239.09 ms     | 5.29 ms         | 45x   |
| maximum size=7  | 242.36 ms     | 4.78 ms         | 51x   |
| minimum size=15 | 1.1553 s      | 4.90 ms         | 236x  |
| maximum size=15 | 905.32 ms     | 8.11 ms         | 112x  |
| minimum size=31 | 3.5518 s      | 8.46 ms         | 420x  |
| maximum size=31 | 3.1620 s      | 8.48 ms         | 373x  |

Naive grows ~O(size^2·log); separable is ~flat, so the ratio grows without bound
with kernel size. (A follow-up could drop the residual constant factor — the
per-element `get_boundary`/`unravel` Vec allocations — with a 1D boundary mapper.)
