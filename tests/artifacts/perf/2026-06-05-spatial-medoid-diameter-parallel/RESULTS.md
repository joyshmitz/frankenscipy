# perf(fsci-spatial): parallelize medoid + diameter all-pairs reductions (byte-identical)

## Lever
medoid (point minimizing sum of distances to all others) and diameter (max pairwise
distance) are O(n^2*d) serial all-pairs reductions. Parallelize across the outer index
(cdist_thread_count gate):
- medoid: each point's total is an independent O(n*d) sum; compute totals in parallel into
  index order, then run the identical serial argmin (strict `<`, lowest index wins ties).
- diameter: row-split max; the combine rule (NaN if any operand NaN, else max) is
  commutative+associative, and `max` selects (no rounding), so the scalar is identical.

## Byte-identity
medoid index + diameter to_bits OLD(serial)==NEW(parallel):
  n=2000 d=8   medoid=1526 diameter=40160a008d5b8a72
  n=4000 d=12  medoid=1330 diameter=4017bcb67120a32a
  n=6000 d=16  medoid=3079 diameter=401ac4fc185f8046

## Bench (perf_medoid, release-perf, min of 3, 64 cores)
| n    | d  | medoid OLD | medoid NEW | x      | diameter OLD | diameter NEW | x      |
|------|----|-----------:|-----------:|-------:|-------------:|-------------:|-------:|
| 2000 | 8  |  16.78 ms  |   4.78 ms  | 3.51x  |   13.01 ms   |   5.17 ms    | 2.52x  |
| 4000 | 12 | 105.42 ms  |   8.62 ms  | 12.23x |   59.18 ms   |   6.07 ms    | 9.75x  |
| 6000 | 16 | 315.45 ms  |  21.13 ms  | 14.93x |  194.71 ms   |  12.94 ms    | 15.05x |
Grows ~n. 182 fsci-spatial tests pass; clippy clean.
