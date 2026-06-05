# fsci-interpolate RbfInterpolator::eval_many: single-threaded -> multithreaded queries

## Target (zero-threaded-crate vein; fsci-interpolate had ZERO threading)

RbfInterpolator::eval_many(queries) mapped each query through eval(q), an O(n_centers)
sum over the RBF centers. Each query is independent -> embarrassingly parallel,
byte-identical. RBF scattered-data interpolation is heavily used.

## Lever (one)

Add a reusable `par_query_map<T: Sync, F: Fn(&T)->f64 + Sync>(queries, work_per_query,
f)` helper: split queries across std::thread::scope workers, each maps its contiguous
range, concatenated in order. Gated on work = queries*centers >= 2^18, cores capped
at queries/2. Applied to RbfInterpolator::eval_many; reusable for the crate's other
per-query batch evaluators (interpn/griddata).

## Isomorphism / proof (BYTE-IDENTICAL)

eval(q) is a pure deterministic sum; each output identical regardless of owning
thread; order preserved. New test rbf_eval_many_parallel_is_bit_identical (2000
queries * 200 centers > gate; Multiquadric/InverseMultiquadric/ThinPlateSpline;
f64::to_bits equal vs sequential map); perf_rbf reports bit_identical=true. fsci-
interpolate 125 passed / 0 failed; lib clippy + fmt clean.

## Same-process A/B (perf_rbf bin, 64-core worker)

| centers | queries | seq | par | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 4000 | 9.47 ms | 4.06 ms | 2.33x |
| 2000 | 6000 | 56.10 ms | 6.33 ms | 8.86x |

Speedup grows with centers (more compute per query). Byte-identical, Score >= 2.0.
Zero-threaded-crate sweep: spatial, ndimage, stats, now interpolate.
