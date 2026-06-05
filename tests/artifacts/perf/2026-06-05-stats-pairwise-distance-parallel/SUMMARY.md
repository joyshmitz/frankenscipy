# fsci-stats pairwise_euclidean_distance_matrix: single-threaded -> multithreaded rows

## Target

pairwise_euclidean_distance_matrix (the O(n^2*d) self-distance matrix used by
multiscale_graphcorr, called 2x) computed the upper triangle in a sequential nested
loop and mirrored. fsci-stats had ~zero threading. Each row's distances are
independent and expensive (d-dim euclidean with sqrt per pair).

## Lever (one)

Rewrite as a FULL row-by-row computation (each row i = [dist(i,0)..dist(i,n-1)],
dist(i,i)=0) and split the rows across std::thread::scope workers. Gated on
work = n*n*d >= 2^18 (n>=8), cores capped at n/2.

## Isomorphism / proof (BYTE-IDENTICAL)

dist(i,j) == dist(j,i) BIT-FOR-BIT in IEEE754 ((a-b)^2 == (b-a)^2; the diagonal is
sqrt(0)=0), so the full row-by-row matrix equals the upper-triangle-and-mirror result
exactly while making every row disjoint/parallel. New test
pairwise_distance_matrix_parallel_is_bit_identical (n=300>gate; f64::to_bits equal vs
a verbatim upper-triangle-mirror reference). 8 multiscale_graphcorr tests pass;
fsci-stats lib.rs clippy-clean (the only -D failure is the fsci-special beta.rs
dependency lint, not this change); fmt clean.

## Function-level A/B (in-test, n=2000 d=16, debug build)

| | seq | par | speedup |
| --- | ---: | ---: | ---: |
| pairwise matrix | 989.9 ms | 69.5 ms | 14.25x |

Same euclidean-cdist kernel already shown to parallelize (cdist 7decfe77, 3.3-6.6x);
the full-matrix row split here is even cleaner (14x). Byte-identical, Score >> 2.0.
