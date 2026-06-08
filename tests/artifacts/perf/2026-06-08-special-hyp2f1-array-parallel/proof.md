# perf: parallelize hyp2f1 hypergeometric vector broadcast loop

Bead: frankenscipy-9ywak

## Lever
`hyp2f1_dispatch` (crates/fsci-special/src/hyper.rs) — behind the Gauss hypergeometric
`hyp2f1(a,b,c,z)` — evaluated its vector broadcast loop (both the real and complex branches)
serially: `for i in 0..out_len { results.push(hyp2f1_scalar(...broadcast at i...)?) }`. Each
output index is an independent series summation (2F1 can run to thousands of terms). Added a
generic index-based `par_map_indices<T>` helper (f64 or Complex64) and routed both broadcast
loops through it.

## Isomorphism / byte-identity argument
- Each output index `i` reads the inputs at broadcast index `i` (`tensor_get_real`/
  `tensor_get_complex`) and computes `hyp2f1_scalar`/`hyp2f1_complex_parameters` written to
  slot `i`; chunks cover `0..out_len` contiguously and concatenate in index order ⇒ identical
  output `Vec`. No reduction.
- Error path: chunk results folded with `?` in chunk (=index) order ⇒ first failing index's
  `SpecialError`, exactly as the serial loop's `?`.
- Empty / broadcast-shape / scalar paths untouched. Gate: serial for `< 64` elements (the
  per-element series is expensive enough that even small arrays amortize).

⇒ The returned value (and first error) is bit-identical to the serial implementation.

## Proof (golden — serial vs parallel, identical)
Harness: `cargo run --profile release-perf -p fsci-special --bin perf_hyp2f1_array`
(a=0.5, b=1.5, c=2.5, z∈(-0.9,0.9))
```
n=100   hyp2f1=8a1e55d0646340be
n=5000  hyp2f1=943757b51c23c404
n=50000 hyp2f1=ea254154d51052b9
timing acc: hyp2f1 200k=6531fa397b8406e1  500k=a1794dc541688169
```
Identical in the stashed serial build and the parallel build.
sha256(golden payload file) = d4371a52d65c6fe8918f5048565c1674b336215de92b5b3a6325070c340cd5e3

## Timing (rch remote, release-perf, 3 back-to-back runs each)
| array | serial (3x)          | new (3x)           | speedup |
|-------|----------------------|--------------------|---------|
| 200k  | 27.38/26.00/27.61 ms | 5.99/6.68/5.82 ms  | ~4.6x   |
| 500k  | 63.08/63.72/62.88 ms | 9.35/9.44/8.13 ms  | ~6.7x   |

(z is in the fast direct-series region here; z near 1 / transformed regions run more terms,
so the real-world ratio is typically higher.)

## Validation
27 hyp2f1 unit tests pass; clippy: no warning in hyper.rs.
