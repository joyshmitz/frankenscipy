# fsci-stats GaussianKde::evaluate_many: single-threaded -> multithreaded points

## Target (bead frankenscipy-n2mqb vein; fsci-stats had ZERO threading)

GaussianKde::evaluate_many(points) mapped each query point through evaluate(x), an
O(n_data) sum over the dataset. Each point is independent (no RNG) -> embarrassingly
parallel, byte-identical. KDE density estimation is heavily used (plotting, stats).

## Lever (one)

Split the query points across threads via std::thread::scope; each thread computes
evaluate(x) for its contiguous point range, results concatenated in order. Gated on
work = points*dataset >= 2^18, cores capped at points/2. The proven row-parallel
pattern (cf. cdist/matmul). GaussianKde is Sync (Vec<f64> + f64), shared by &.

## Isomorphism / proof (BYTE-IDENTICAL)

evaluate(x) is a pure deterministic sum; each output point is identical regardless of
the owning thread; order preserved by concatenating contiguous chunks. New test
gaussian_kde_evaluate_many_parallel_is_bit_identical (2000 pts * 400 data > gate,
f64::to_bits equal vs the sequential map); perf_kde reports bit_identical=true at
scale. fsci-stats lib tests pass; the crate's own lib.rs is clippy-clean (the only
-D-warnings failure is a PRE-EXISTING if_same_then_else lint in the fsci-special
dependency, beta.rs:1181 — not this change).

## Same-process A/B (perf_kde bin, debug build — release shared-target cache was
## rustc-skewed for rand deps; the seq/par RATIO is valid and conservative)

| n_data | n_points | seq | par | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | 4000 | 50.17 ms | 5.90 ms | 8.51x |
| 5000 | 8000 | 496.42 ms | 24.25 ms | 20.47x |

Byte-identical, Score >> 2.0. fsci-stats (largest crate) had zero threading — KDE is
the first parallel kernel; bootstrap/permutation (RNG, need per-index seeding) next.
