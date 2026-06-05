# perf(fsci-stats): multiscale_graphcorr mgc_map O(n^4) -> O(n^2) (frankenscipy-9tnvb)

## Lever
`compute_mgc_map` looped all n^2 scales (k,l), and each `local_correlation` did a full
O(n^2) pass over (i,j) pairs -> O(n^4), intractable at the harness sizes (n=1500/2500).

A pair (i,j) with ranks (a,b)=(rank_x[i][j],rank_y[i][j]) is included in local_corr(k,l)
for EVERY scale with k>a and l>b. So scatter each pair's {cx*cy, cx*cx, cy*cy, 1} into
bucket (a,b) of four accumulator grids, then take a 2D inclusive prefix sum; the prefix
value at (k-1,l-1) is exactly the sum over a<k, b<l. Whole map in O(n^2).

## Isomorphism proof
Unit test `multiscale_graphcorr_mgc_map_matches_on4_reference` (n=12,37,64) asserts the
prefix-sum map matches the retained O(n^4) `local_correlation` oracle to max-abs < 1e-12.
Tolerance-parity (not bit-exact): within a bucket the products are summed in the original
(i,j) row-major order, so each bucket equals the original partial sum; only cross-bucket
recombination order differs. Two cumulative passes (no inclusion-exclusion subtraction)
keep it numerically stable. All 9 multiscale_graphcorr tests pass; clippy clean.

## Bench (perf_mgc, release-perf, ms/run, same machine)
| n   | OLD O(n^4) | NEW O(n^2) | speedup |
|-----|-----------:|-----------:|--------:|
| 120 |    287.2   |     1.6    |   180x  |
| 250 |   5482.5   |    14.0    |   392x  |
| 400 |  35249.5   |    23.3    |  1513x  |
Speedup grows ~O(n^2). At harness n=1500/2500 the OLD map is ~hours; NEW whole pipeline
runs 350 ms / 1052 ms (now bounded by O(n^2 log n) row ranks + O(n^2 d) distances).
