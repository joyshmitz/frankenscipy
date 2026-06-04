# medfilt: per-window O(k) selection → sliding two-multiset median (O(n·log k))

Bench: `medfilt` (fsci-signal), rch ts2. Function: `medfilt`.

## Lever
The 1D median filter refilled a length-k window and ran `select_nth_unstable_by`
per output — O(n·k). For large kernels it now uses a sliding median built from
two `total_cmp`-ordered `BTreeMap` multisets: `lower` keeps the smallest
ceil(total/2) values (its maximum is the median), `upper` the rest, with
`max(lower) <= min(upper)`. Each window step removes the leaving value and
inserts the entering value (zero-padded at the borders) in O(log k), giving
O(n·log k). Gated at kernel_size >= 64; smaller kernels keep the naive path
(faster there).

## Isomorphism
The median is `lower`'s maximum = the rank-(k/2) value by `total_cmp`, exactly
what `select_nth_unstable_by(k/2)` returns (k is always odd). Removal can take
any copy of the leaving value because the median depends only on the union and
rebalance restores the size invariant. Proven by
`medfilt_sliding_matches_naive_loop`: `.to_bits()` equality vs the per-window
select reference across n in {1,50,400,1000}, k in {1,3,63,65,101,257}, tie
densities from continuous to heavily tied (the multiset-straddle case), and
signed zeros. Existing medfilt tests pass; clippy + fmt clean.

## Benchmark (8192-sample signal, rch ts2, naive vs sliding)
| case            | naive     | sliding  | Score |
|-----------------|-----------|----------|-------|
| medfilt 8192 k65  | 2.851 ms  | 1.803 ms | 1.58x |
| medfilt 8192 k257 | 9.557 ms  | 2.713 ms | 3.52x |
| medfilt 8192 k513 | 16.181 ms | 3.105 ms | 5.22x |

Naive scales O(n·k); sliding O(n·log k), so the ratio grows with kernel size.
(k65 is near the crossover — still a win; the big wins are at the larger kernels
where the linear-in-k cost dominates.)
