# perf: parallelize polygamma + rgamma real-array dispatch

Bead: frankenscipy-com3q

## Lever
`polygamma(n, z)` (n-th derivative of digamma — Bayesian Fisher-information / log-gamma
variance terms; higher orders use an expensive Hurwitz-zeta-like series) and `rgamma`
(reciprocal gamma 1/Γ) in crates/fsci-special/src/gamma.rs mapped their per-element kernels
serially over the `RealVec` arm (polygamma also `ComplexVec`). Routed through the
`par_map_indices` helper already in gamma.rs. For polygamma the per-index kernel selects on
`n` (digamma/trigamma/tetragamma/polygamma_higher).

## Isomorphism / byte-identity argument
- Each output index `i` is the same scalar kernel of `values[i]` written to slot `i`; chunks
  cover `0..n` contiguously and concatenate in index order ⇒ identical output `Vec`. No
  reduction.
- Error path: chunk results folded with `?` in chunk (=index) order ⇒ first failing index's
  `SpecialError`, exactly as the serial `.iter().map(kernel).collect()`.
- Scalar / Empty / (rgamma) complex arms untouched. Gate: serial for `< 256` elements.

⇒ The returned value (and first error) is bit-identical to the serial implementation.

## Proof (golden — serial vs parallel, identical)
Harness: `cargo run --profile release-perf -p fsci-special --bin perf_polygamma_array`
(polygamma order 5, x∈(0.5,20.5))
```
n=100   polygamma5=e81d4790b5314f43 rgamma=245db3f74313944c
n=5000  polygamma5=668adf29e28be6dd rgamma=41569423053d98da
n=50000 polygamma5=febf0c24541a1785 rgamma=257eca53ce159e84
timing acc: polygamma5 1M=28db5cf7e0471fba  rgamma 4M=0120f3ed128e33ef
```
Identical in the stashed serial build and the parallel build.
sha256(golden payload file) = c0b1b74ac9bfa6d0b1c2a4de2b3c533409fd51601569590572c5bd2f3a39942a

## Timing (rch remote, release-perf, 3 back-to-back runs each)
| function / array  | serial (3x)             | new (3x)             | speedup |
|-------------------|-------------------------|----------------------|---------|
| polygamma(5), 1M  | 253.4/262.2/258.6 ms    | 17.60/17.47/17.03 ms | ~14.7x  |
| rgamma,       4M  | 142.96/146.06/154.15 ms | 20.20/19.91/22.60 ms | ~7.2x   |

## Validation
11 polygamma/rgamma unit tests pass; clippy: no warning in gamma.rs.
