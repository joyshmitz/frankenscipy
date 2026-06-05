# fsci-spatial cdist: single-threaded pairwise distances -> multithreaded row split

## Target

cdist_metric (and cdist / distance_matrix which call it) computed the full na×nb
pairwise distance matrix with a single-threaded nested map. fsci-spatial had ZERO
threading. Each output row (distances from xa[i] to all of xb) is an independent
reduction over the pure `metric_distance` — embarrassingly parallel, byte-identical.

## Lever (one)

Split the rows of `xa` across threads via std::thread::scope; each thread computes a
disjoint contiguous range of output rows and the results are concatenated in order.
Gated by `cdist_thread_count` (work = na·nb·dim): sequential below 2^18, else cores
capped at na/2. The proven no-tokio row-parallel pattern (cf. matmul/inv/cholesky).

## Isomorphism / proof (BYTE-IDENTICAL)

Every output[i][j] = metric_distance(xa[i], xb[j], metric), the identical computation
regardless of which thread runs row i; row order is preserved by concatenating thread
chunks in index order. New test cdist_metric_parallel_is_bit_identical (na=nb=600 >
gate, metrics Euclidean/Cityblock/Chebyshev, f64::to_bits equal). perf_cdist bin
reports bit_identical=true at scale. fsci-spatial 181 passed / 0 failed; lib clippy +
fmt clean.

## Same-process A/B (perf_cdist bin, 64-core worker)

| na | nb | dim | seq | par | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 2000 | 3 | 28.50 ms | 8.75 ms | 3.26x |
| 4000 | 1000 | 8 | 33.47 ms | 8.55 ms | 3.92x |
| 3000 | 3000 | 16 | 109.96 ms | 16.70 ms | 6.59x |

Speedup grows with dim (more compute per entry amortises the Vec<Vec> allocation +
memory traffic). Byte-identical, Score >> 2.0. BROAD VEIN: pdist (condensed, needs
index mapping), and fsci-ndimage filters (also zero-threaded) are next.
