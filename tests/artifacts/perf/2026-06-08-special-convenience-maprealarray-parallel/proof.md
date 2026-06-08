# perf: parallelize convenience-module real-array dispatch (convenience::map_real)

Bead: frankenscipy-nlzc1

## Lever
`map_real` (crates/fsci-special/src/convenience.rs) is the shared unary real-array
dispatcher behind **30 convenience functions** — including `ndtr`/`ndtri` (standard normal
CDF and probit/quantile — among the most ubiquitous functions in all of stats & ML),
`log_ndtr`, `spence` (dilogarithm), `kolmogorov`/`kolmogi`, and the ML activations
`expit`/`logit`/`softplus`/`mish`/`silu`/`log_cosh`/`softsign`/`hard_sigmoid`/... It mapped
its per-element kernel serially over the `RealVec` arm. Added a generic index-based helper
`par_map_indices` and routed that arm through it.

## Isomorphism / byte-identity argument
- Each output index `i` is `kernel(values[i])` written to slot `i`; chunks cover `0..n`
  contiguously and concatenate in index order ⇒ identical output `Vec`. No reduction.
- Error path: chunk results folded with `?` in chunk (=index) order ⇒ first failing index's
  `SpecialError`, exactly as the serial `values.iter().map(kernel).collect()`.
- Scalar / unsupported-type arms untouched. Gate: serial for `< 256` elements.

⇒ The returned value (and first error) is bit-identical to the serial implementation.

## Proof (golden — serial vs parallel, identical)
Harness: `cargo run --profile release-perf -p fsci-special --bin perf_ndtr_array`
```
n=100   ndtr=8bfb07c9c46b5271 ndtri=2e686b7119b5bdae
n=5000  ndtr=f90abcdccfe51dec ndtri=158d14cb8905e1eb
n=50000 ndtr=e7918ab1b53ec289 ndtri=5515e7e68d558524
timing acc: ndtr 4M=59874b0ec7cd990e  ndtri 2M=5978a5a4dee5586a
```
Identical in the stashed serial build and the parallel build.
sha256(golden payload file) = 67408eba8608aa9ede9f2e6c0e1d4be1cc04e54bf11bd3b466a44b7ae4b8845b

## Timing (rch remote, release-perf, 3 back-to-back runs each)
| function / array | serial (3x)             | new (3x)             | speedup |
|------------------|-------------------------|----------------------|---------|
| ndtr,  4M        | 1.026/1.005/1.003 s     | 44.54/45.55/50.22 ms | ~22.1x  |
| ndtri, 2M        | 1.252/1.272/1.267 s     | 45.63/45.58/46.98 ms | ~27.8x  |

## Validation
28 ndtr/convenience unit tests pass; clippy: no warning in convenience.rs (helper + arm clean).
