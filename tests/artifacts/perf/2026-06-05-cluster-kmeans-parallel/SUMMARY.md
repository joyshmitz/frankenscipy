# fsci-cluster kmeans: single-threaded assignment -> multithreaded, byte-identical

## Target (zero-threaded-crate vein; fsci-cluster had ZERO threading)

kmeans' per-iteration assignment step (each point -> nearest of k centroids,
O(k*d) per point) ran in a sequential loop. Each point's nearest_centroid is
independent and moderately expensive (k*d squared-distance ops).

## Lever (one)

Add assign_points: compute (label, sq_dist) per point in parallel across
std::thread::scope workers, then the caller sums inertia SEQUENTIALLY in point order.
Gated high (work = n*k*d >= 2^21) with nthreads capped at n/32 to avoid oversubscribing
on this moderate-cost-per-point kernel (the barycentric over-trigger lesson).

## Isomorphism / proof (BYTE-IDENTICAL)

nearest_centroid is pure and deterministic, so each (label, sq_dist) is identical
regardless of thread; labels written in order; CRUCIALLY the inertia float reduction
stays sequential in point order, so the convergence check and iteration count are
unchanged -> the whole kmeans result (labels, centroids, inertia, n_iter) is identical.
Proven by stash A/B: result digest (FNV over labels + inertia.to_bits + n_iter)
UNCHANGED: n=20000 b5184a0ce5480389, n=40000 a3c34a0fb647e9d2. fsci-cluster tests
pass; clippy + fmt clean.

## Rebench (perf_kmeans, deterministic data, same worker via stash)

| n | k | d | before (seq) | after (parallel) | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20000 | 50 | 16 | 168.79 ms | 79.66 ms | 2.12x |
| 40000 | 80 | 24 | 581.14 ms | 124.84 ms | 4.66x |

Win grows with k*d. Byte-identical, Score >= 2.0. 5th zero-threaded crate parallelized
(spatial, ndimage, stats, interpolate, cluster).
