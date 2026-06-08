# perf: parallelize complex beta/betaln array arms

Bead: frankenscipy-f143c

## Lever
`map_complex_binary` (crates/fsci-special/src/beta.rs) — the complex binary tensor dispatcher
behind `beta` (B(a,b) = Γ(a)Γ(b)/Γ(a+b)) and `betaln` (log|B|). Its **8 complex array arms**
(Complex×Complex and mixed Real/Complex; vec×scalar, scalar×vec, vec×vec for each) mapped the
infallible `complex_beta_scalar` / `complex_betaln_scalar` kernel serially with
`.iter().map(...).collect()`. Each kernel is three complex log-Gamma evaluations, so it is
moderately expensive. Made the file's `par_map_indices` helper generic over `T: Send` (it was
f64-only), added `+ Sync` to the `F` kernel bound, and routed all 8 complex array arms through
`par_map_indices` (wrapping the infallible kernel as `|i| Ok(kernel(...))`). Scalar arms,
length checks, and the fail-closed arm are untouched.

## Isomorphism / byte-identity argument
- `complex_*_scalar` is a pure infallible `fn(Complex64, Complex64) -> Complex64`. Each output
  index `i` is `kernel(...broadcast at i...)` written to slot `i`; for mixed arms the real
  operand is lifted with `Complex64::from_real` at the same index. Chunks cover the index range
  contiguously and concatenate in index order ⇒ identical output `Vec<Complex64>`. No reduction.
- The kernel never returns `Err`, so the `Ok(...)` wrapping is total and `par_map_indices`
  reproduces the exact element order; vec×vec length checks preserved; gate serial for `< 256`.

⇒ The returned value is bit-identical to the serial implementation.

## Proof (golden — serial vs parallel, identical)
Harness: `cargo run --profile release-perf -p fsci-special --bin perf_beta_complex`
(b = 1.7+0.4i complex scalar, a complex vec re∈(0.5,6.5) im∈(-2,2))
```
n=100 beta(len=100,xor=214d3c2994c73fea) betaln(len=100,xor=b4086d8407d83da0)
n=5000 beta(len=5000,xor=88b847ee7ec1be78) betaln(len=5000,xor=e09255e0a956c84c)
n=50000 beta(len=50000,xor=1d6a695a45c388a4) betaln(len=50000,xor=c5983ad07db5a53d)
timing acc: beta 500k=23a0e811045ed513  betaln 500k=eafb4859602342e3
```
Identical in the stashed serial build and the parallel build (golden xor + timing acc match).
sha256(golden_payload.txt) = 296f39931ddff08b67b3c6775d24d5d3e631d4adb93b8b80928cadf21edc981d

## Timing (rch remote, release-perf, 3 back-to-back runs each)
| function / array | serial (3x)             | new (3x)               | speedup |
|------------------|-------------------------|------------------------|---------|
| beta, 500k cplx  | 107.2/112.5/107.7 ms    | 11.45/13.74/14.31 ms   | ~8.4x   |
| betaln, 500k cplx| 103.8/96.7/99.1 ms      | 9.36/8.82/9.58 ms      | ~11x    |

## Validation
74 fsci-special beta-related unit tests pass; clippy: no warning in beta.rs.
