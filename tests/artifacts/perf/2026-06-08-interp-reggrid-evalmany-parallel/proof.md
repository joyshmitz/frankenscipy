# perf: parallelize RegularGridInterpolator::eval_many over the query batch

Bead: frankenscipy-h6ts5

## Lever
`RegularGridInterpolator::eval_many` (crates/fsci-interpolate/src/lib.rs) ran a serial
`xi.iter().map(|x| self.eval(x)).collect()`. Each query is an independent evaluation
(read-only interval lookup + interpolation over the shared grid; no mutable state), so
route the batch through the crate's existing `par_query_try_map` helper (already used by
`RbfInterpolator`/`LinearNDInterpolator`) with a per-method work estimate.

## Isomorphism / byte-identity argument
`par_query_try_map` splits the queries into contiguous chunks, evaluates the same pure
`eval` on each, then concatenates the chunk results in order; on error it returns the
first failing chunk's error in query order. So both the success Vec and the error are
bit-identical to the sequential `map(...).collect()`. No reduction is performed (each
output is an independent scalar written to its own slot). The Nearest+3D fast path is
untouched. Gate: stays serial for `work < 2^18` or `< 4` queries.

## Proof (golden — serial baseline vs parallel NEW, identical)
Harness: `cargo run --profile release-perf -p fsci-interpolate --bin perf_reggrid_eval`
(Cubic method, deterministic grid + queries.)

```
ndim=2 len=40 m=1000 out_xor_bits=89216ecadcfba55e n_out=1000
ndim=3 len=24 m=5000 out_xor_bits=bff3b70b4aee2ccb n_out=5000
```
Identical bits in the stashed serial build and the parallel build.
sha256(golden serial baseline payload) =
4347327ab9955e2e29eebaf5d57eb964102cd2ce0376ec9069bfba287c43db34

## Timing (rch remote, release-perf, 3 back-to-back runs each — Cubic)
| case                  | serial (3x)        | new (3x)         | speedup |
|-----------------------|--------------------|------------------|---------|
| ndim=3 len=30 m=200k  | 88.4/90.1/95.4 ms  | 8.1/8.9/9.3 ms   | ~10.3x  |
| ndim=4 len=16 m=200k  | 441/445/453 ms     | 22.8/23.3/23.9 ms| ~19.1x  |

NOTE: single-shot measurements on the (contended) remote were noisy — an early NEW
sample read 59 ms where the back-to-back median is ~9 ms. The 3x back-to-back runs above
are the reliable figures. Lesson: bench 3x back-to-back on a shared remote.

## Validation
17 regular_grid unit tests pass; clippy clean (the one warning is in the fsci-fft dep).
